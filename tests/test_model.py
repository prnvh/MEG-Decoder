import unittest

import torch

from general.bdh import BDHConfig, ContinuousBDH, count_trainable_parameters


class ContinuousBDHTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = BDHConfig(
            d_model=8,
            n_neurons=32,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            state_decay=1.0,
        )
        self.model = ContinuousBDH(self.config).eval()

    def assert_tensors_close(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_recurrent_matches_parallel_oracle(self) -> None:
        inputs = torch.randn(2, 7, self.config.d_model)

        recurrent, _, _ = self.model(inputs)
        parallel = self.model.forward_parallel(inputs)

        self.assert_tensors_close(recurrent, parallel)

    def test_chunk_continuation_matches_one_pass(self) -> None:
        inputs = torch.randn(2, 9, self.config.d_model)

        one_pass, one_pass_state, _ = self.model(inputs)
        first, state, _ = self.model(inputs[:, :4])
        second, chunked_state, _ = self.model(inputs[:, 4:], state=state)

        self.assert_tensors_close(torch.cat((first, second), dim=1), one_pass)
        for actual, expected in zip(
            chunked_state.layers, one_pass_state.layers, strict=True
        ):
            self.assert_tensors_close(actual, expected)
        self.assertTrue(torch.equal(chunked_state.position, one_pass_state.position))

    def test_masked_tail_does_not_read_or_update_state(self) -> None:
        prefix = torch.randn(2, 4, self.config.d_model)
        padded = torch.cat(
            (prefix, torch.randn(2, 3, self.config.d_model)), dim=1
        )
        mask = torch.tensor(
            [[True, True, True, True, False, False, False]] * 2
        )

        prefix_output, prefix_state, _ = self.model(prefix)
        padded_output, padded_state, _ = self.model(padded, mask)

        self.assert_tensors_close(padded_output[:, :4], prefix_output)
        self.assertTrue(torch.count_nonzero(padded_output[:, 4:]) == 0)
        for actual, expected in zip(
            padded_state.layers, prefix_state.layers, strict=True
        ):
            self.assert_tensors_close(actual, expected)
        self.assertTrue(torch.equal(padded_state.position, prefix_state.position))

    def test_reset_mask_clears_only_selected_streams(self) -> None:
        history = torch.randn(2, 5, self.config.d_model)
        current = torch.randn(2, 3, self.config.d_model)
        _, history_state, _ = self.model(history)

        reset_output, reset_state, _ = self.model(
            current,
            state=history_state,
            reset_mask=torch.tensor([True, False]),
        )
        fresh_output, fresh_state, _ = self.model(current[:1])

        self.assert_tensors_close(reset_output[:1], fresh_output)
        for actual, expected in zip(
            reset_state.layers, fresh_state.layers, strict=True
        ):
            self.assert_tensors_close(actual[:1], expected)
        self.assertEqual(reset_state.position.tolist(), [3, 8])

    def test_detach_state_stops_cross_chunk_gradients(self) -> None:
        first_inputs = torch.randn(
            1, 4, self.config.d_model, requires_grad=True
        )
        second_inputs = torch.randn(
            1, 3, self.config.d_model, requires_grad=True
        )
        _, state, _ = self.model(first_inputs)

        second_output, _, _ = self.model(
            second_inputs, state=state, detach_state=True
        )
        second_output.square().mean().backward()

        self.assertIsNone(first_inputs.grad)
        self.assertIsNotNone(second_inputs.grad)
        self.assertTrue(torch.isfinite(second_inputs.grad).all())

    def test_future_inputs_do_not_change_past_outputs(self) -> None:
        inputs = torch.randn(1, 8, self.config.d_model)
        changed = inputs.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 20.0

        output, _, _ = self.model(inputs)
        changed_output, _, _ = self.model(changed)

        self.assert_tensors_close(output[:, :4], changed_output[:, :4])

    def test_parameter_count_uses_three_shared_matrices(self) -> None:
        expected = 3 * self.config.n_neurons * self.config.d_model
        self.assertEqual(count_trainable_parameters(self.model), expected)

    def test_backward_produces_finite_parameter_gradients(self) -> None:
        self.model.train()
        inputs = torch.randn(2, 5, self.config.d_model)
        output, _, _ = self.model(inputs)
        output.square().mean().backward()

        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
