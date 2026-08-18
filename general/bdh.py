"""Continuous-input BDH with explicit recurrent attention state.

The public BDH implementation is written for token IDs and materializes a
``T x T`` causal attention matrix.  This model accepts already-encoded
continuous inputs with shape ``[B, T, D]`` and represents linear attention as
an incrementally updated state.  It can therefore continue across chunks
without retaining or recomputing the full input history.

MEG channel encoding and task-specific prediction heads intentionally live
outside this module.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclasses.dataclass(frozen=True)
class BDHConfig:
    """Configuration for the latent-input BDH core."""

    d_model: int = 128
    n_neurons: int = 2048
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    state_decay: float = 1.0
    rope_theta: float = float(2**16)

    def __post_init__(self) -> None:
        positive_ints = {
            "d_model": self.d_model,
            "n_neurons": self.n_neurons,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.n_neurons % self.n_heads != 0:
            raise ValueError("n_neurons must be divisible by n_heads")
        if self.neurons_per_head % 2 != 0:
            raise ValueError("n_neurons / n_heads must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.state_decay <= 1.0:
            raise ValueError("state_decay must be in (0, 1]")
        if self.rope_theta <= 0.0:
            raise ValueError("rope_theta must be positive")

    @property
    def neurons_per_head(self) -> int:
        return self.n_neurons // self.n_heads


@dataclasses.dataclass(frozen=True)
class BDHState:
    """Persistent state for a batch of continuous streams.

    Every layer state has shape ``[B, H, N_h, D]``. ``position`` counts valid
    input steps independently for each batch item so RoPE remains continuous
    when a recording is split into chunks.
    """

    layers: tuple[Tensor, ...]
    position: Tensor

    def detach(self) -> "BDHState":
        """Stop gradients through history without clearing numeric state."""

        return BDHState(
            layers=tuple(layer.detach() for layer in self.layers),
            position=self.position.detach(),
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "BDHState":
        """Move floating state while keeping positions integer-valued."""

        return BDHState(
            layers=tuple(
                layer.to(device=device, dtype=dtype) for layer in self.layers
            ),
            position=self.position.to(device=device),
        )


def _frequencies(size: int, theta: float) -> Tensor:
    """Pair-quantized oscillator frequencies used by the BDH reference."""

    coordinates = torch.arange(size, dtype=torch.float32)
    paired_coordinates = torch.floor(coordinates / 2.0) * 2.0
    return 1.0 / (theta ** (paired_coordinates / size)) / (2.0 * math.pi)


class ContinuousBDH(nn.Module):
    """BDH core for continuous latent sequences.

    The three learned matrices are shared across all recurrent layers, matching
    the reference architecture.  Attention state is separated by layer and
    head.  The current input reads ``rho_(t-1)`` before its key/value update is
    written, making the operation strictly causal.
    """

    def __init__(self, config: BDHConfig) -> None:
        super().__init__()
        self.config = config
        h = config.n_heads
        d = config.d_model
        n = config.neurons_per_head

        # Paper notation: two D -> N decoders and one N -> D encoder.
        self.decoder_x = nn.Parameter(torch.empty(h, d, n))
        self.decoder_y = nn.Parameter(torch.empty(h, d, n))
        self.encoder = nn.Parameter(torch.empty(h, n, d))

        self.input_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.attention_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.residual_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "frequencies",
            _frequencies(n, config.rope_theta).view(1, 1, n),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.decoder_x, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_y, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder, mean=0.0, std=0.02)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> BDHState:
        """Create an empty state suitable for ``batch_size`` streams."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (
            batch_size,
            self.config.n_heads,
            self.config.neurons_per_head,
            self.config.d_model,
        )
        return BDHState(
            layers=tuple(
                torch.zeros(shape, device=device, dtype=dtype)
                for _ in range(self.config.n_layers)
            ),
            position=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def _validate_state(self, state: BDHState, inputs: Tensor) -> None:
        batch_size = inputs.shape[0]
        expected_shape = (
            batch_size,
            self.config.n_heads,
            self.config.neurons_per_head,
            self.config.d_model,
        )
        if len(state.layers) != self.config.n_layers:
            raise ValueError(
                f"state has {len(state.layers)} layers, expected "
                f"{self.config.n_layers}"
            )
        for index, layer in enumerate(state.layers):
            if layer.shape != expected_shape:
                raise ValueError(
                    f"state layer {index} has shape {tuple(layer.shape)}, "
                    f"expected {expected_shape}"
                )
            if layer.device != inputs.device or layer.dtype != inputs.dtype:
                raise ValueError(
                    "state tensors must match the input device and dtype"
                )
        if state.position.shape != (batch_size,):
            raise ValueError(
                f"state.position must have shape [{batch_size}], "
                f"got {tuple(state.position.shape)}"
            )
        if state.position.device != inputs.device:
            raise ValueError("state.position must be on the input device")

    def _prepare_state(
        self,
        inputs: Tensor,
        state: BDHState | None,
        reset_mask: Tensor | None,
        detach_state: bool,
    ) -> BDHState:
        batch_size = inputs.shape[0]
        if state is None:
            state = self.initial_state(
                batch_size, device=inputs.device, dtype=inputs.dtype
            )
        else:
            self._validate_state(state, inputs)
            if detach_state:
                state = state.detach()

        if reset_mask is None:
            return state
        if reset_mask.shape != (batch_size,) or reset_mask.dtype is not torch.bool:
            raise ValueError("reset_mask must be boolean with shape [B]")
        if reset_mask.device != inputs.device:
            raise ValueError("reset_mask must be on the input device")

        keep = (~reset_mask).to(dtype=inputs.dtype).view(batch_size, 1, 1, 1)
        return BDHState(
            layers=tuple(layer * keep for layer in state.layers),
            position=torch.where(
                reset_mask, torch.zeros_like(state.position), state.position
            ),
        )

    def _rope(self, activations: Tensor, position: Tensor) -> Tensor:
        phases = (
            position.to(dtype=self.frequencies.dtype).view(-1, 1, 1)
            * self.frequencies
        )
        phases = (phases % 1.0) * (2.0 * math.pi)
        rotated = torch.stack(
            (-activations[..., 1::2], activations[..., ::2]), dim=-1
        ).reshape_as(activations)
        return (
            activations * torch.cos(phases).to(dtype=activations.dtype)
            + rotated * torch.sin(phases).to(dtype=activations.dtype)
        )

    def _decode_x(self, values: Tensor) -> Tensor:
        return F.relu(torch.einsum("bd,hdn->bhn", values, self.decoder_x))

    def _decode_y(self, values: Tensor) -> Tensor:
        return F.relu(torch.einsum("bhd,hdn->bhn", values, self.decoder_y))

    def _encode(self, activations: Tensor) -> Tensor:
        return torch.einsum("bhn,hnd->bd", activations, self.encoder)

    def forward(
        self,
        inputs: Tensor,
        sample_mask: Tensor | None = None,
        *,
        state: BDHState | None = None,
        reset_mask: Tensor | None = None,
        detach_state: bool = False,
    ) -> tuple[Tensor, BDHState, dict[str, Tensor]]:
        """Process one latent chunk in chronological order.

        Args:
            inputs: Continuous latent values with shape ``[B, T, D]``.
            sample_mask: Boolean valid-step mask with shape ``[B, T]``. Invalid
                steps neither read nor update state and produce zero output.
            state: State returned by the preceding chunk of the same streams.
            reset_mask: Boolean ``[B]`` mask clearing selected incoming states.
            detach_state: Detach incoming history for truncated BPTT.

        Returns:
            Hidden sequence ``[B, T, D]``, next state, and scalar diagnostics.
        """

        if inputs.ndim != 3 or inputs.shape[-1] != self.config.d_model:
            raise ValueError(
                f"inputs must have shape [B, T, {self.config.d_model}], "
                f"got {tuple(inputs.shape)}"
            )
        batch_size, time_steps, _ = inputs.shape
        if batch_size <= 0:
            raise ValueError("batch dimension must be non-empty")
        if sample_mask is None:
            sample_mask = torch.ones(
                batch_size, time_steps, device=inputs.device, dtype=torch.bool
            )
        if sample_mask.shape != inputs.shape[:2] or sample_mask.dtype is not torch.bool:
            raise ValueError("sample_mask must be boolean with shape [B, T]")
        if sample_mask.device != inputs.device:
            raise ValueError("sample_mask must be on the input device")

        current_state = self._prepare_state(
            inputs, state, reset_mask, detach_state
        )
        layer_states = list(current_state.layers)
        position = current_state.position
        normalized_inputs = self.input_norm(inputs)
        outputs: list[Tensor] = []

        active_counts = inputs.new_zeros(self.config.n_layers)
        possible_counts = inputs.new_zeros(self.config.n_layers)

        for time_index in range(time_steps):
            valid = sample_mask[:, time_index]
            valid_value = valid.view(batch_size, 1)
            valid_state = valid.view(batch_size, 1, 1, 1)
            value = normalized_inputs[:, time_index]

            for layer_index in range(self.config.n_layers):
                sparse_x = self._decode_x(value)
                query_key = self._rope(sparse_x, position)
                rho_previous = layer_states[layer_index]

                # Strict causal read: the current key/value is absent here.
                attended = torch.einsum(
                    "bhn,bhnd->bhd", query_key, rho_previous
                )
                attended = self.attention_norm(attended)
                sparse_y = self._decode_y(attended)
                gated = self.dropout(sparse_x * sparse_y)
                residual = self.residual_norm(self._encode(gated))
                next_value = self.input_norm(value + residual)

                update = torch.einsum("bhn,bd->bhnd", query_key, value)
                rho_next = self.config.state_decay * rho_previous + update
                layer_states[layer_index] = torch.where(
                    valid_state, rho_next, rho_previous
                )
                value = torch.where(
                    valid_value, next_value, torch.zeros_like(next_value)
                )

                active_counts[layer_index] += (
                    (sparse_x > 0)
                    & valid.view(batch_size, 1, 1)
                ).sum().to(dtype=inputs.dtype)
                possible_counts[layer_index] += (
                    valid.sum().to(dtype=inputs.dtype)
                    * float(self.config.n_neurons)
                )

            outputs.append(value)
            position = position + valid.to(dtype=position.dtype)

        if outputs:
            output = torch.stack(outputs, dim=1)
        else:
            output = inputs.new_zeros(batch_size, 0, self.config.d_model)

        next_state = BDHState(layers=tuple(layer_states), position=position)
        diagnostics = {
            "activation_sparsity": (
                1.0 - active_counts / possible_counts.clamp_min(1.0)
            ).detach(),
            "state_norm": torch.stack(
                [layer.detach().float().norm() for layer in next_state.layers]
            ),
            "position": position.detach().clone(),
        }
        return output, next_state, diagnostics

    def forward_parallel(self, inputs: Tensor) -> Tensor:
        """Explicit short-sequence oracle for implementation tests.

        This method intentionally allocates pairwise ``T x T`` scores. It is
        provided only to verify the recurrent implementation and MUST NOT be
        used as the long-context inference path.
        """

        if self.config.state_decay != 1.0:
            raise ValueError("parallel oracle requires state_decay == 1")
        if inputs.ndim != 3 or inputs.shape[-1] != self.config.d_model:
            raise ValueError(
                f"inputs must have shape [B, T, {self.config.d_model}]"
            )

        _, time_steps, _ = inputs.shape
        values = self.input_norm(inputs)
        positions = torch.arange(time_steps, device=inputs.device)

        for _ in range(self.config.n_layers):
            sparse_x = F.relu(
                torch.einsum("btd,hdn->bhtn", values, self.decoder_x)
            )
            phases = (
                positions.to(self.frequencies.dtype).view(1, 1, time_steps, 1)
                * self.frequencies.unsqueeze(2)
            )
            phases = (phases % 1.0) * (2.0 * math.pi)
            rotated = torch.stack(
                (-sparse_x[..., 1::2], sparse_x[..., ::2]), dim=-1
            ).reshape_as(sparse_x)
            query_key = (
                sparse_x * torch.cos(phases).to(dtype=sparse_x.dtype)
                + rotated * torch.sin(phases).to(dtype=sparse_x.dtype)
            )
            scores = torch.einsum("bhtn,bhsn->bhts", query_key, query_key)
            scores = scores.tril(diagonal=-1)
            attended = torch.einsum("bhts,bsd->bhtd", scores, values)
            attended = self.attention_norm(attended)
            sparse_y = F.relu(
                torch.einsum("bhtd,hdn->bhtn", attended, self.decoder_y)
            )
            gated = self.dropout(sparse_x * sparse_y)
            residual = self.residual_norm(
                torch.einsum("bhtn,hnd->btd", gated, self.encoder)
            )
            values = self.input_norm(values + residual)
        return values


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number used for parameter-matched comparisons."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# Short, discoverable alias for callers that do not need to distinguish this
# implementation from the token-input reference model.
BDH = ContinuousBDH
