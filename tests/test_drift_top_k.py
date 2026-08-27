import unittest

import torch

from drifting_core.imagenet_loss import (
    _accumulate_weighted_targets,
    _column_top_k_preserve_mass,
    _top_k_preserve_mass,
    _truncate_force_group,
    drift_loss_imagenet_mixed,
)


class DriftTopKTest(unittest.TestCase):
    def test_top_k_is_rowwise_and_preserves_each_rows_mass(self):
        weights = torch.tensor(
            [
                [
                    [0.60, 0.10, 0.20, 0.10],
                    [0.05, 0.70, 0.10, 0.15],
                ]
            ],
            dtype=torch.float32,
        )
        kept, indices = _top_k_preserve_mass(weights, top_k=2)
        self.assertIsNotNone(indices)
        self.assertEqual(set(indices[0, 0].tolist()), {0, 2})
        self.assertEqual(set(indices[0, 1].tolist()), {1, 3})
        torch.testing.assert_close(kept.sum(2), weights.sum(2))

    def test_zero_and_full_top_k_return_original_dense_tensor(self):
        weights = torch.rand(2, 3, 5)
        for top_k in (0, 5, 8):
            actual, indices = _top_k_preserve_mass(weights, top_k=top_k)
            self.assertIs(actual, weights)
            self.assertIsNone(indices)

    def test_forward_top_k_selects_generated_rows_per_target_column(self):
        # Columns deliberately prefer different generated rows.
        weights = torch.tensor(
            [
                [
                    [0.70, 0.10, 0.20, 0.10],
                    [0.20, 0.80, 0.30, 0.20],
                    [0.10, 0.10, 0.50, 0.70],
                ]
            ],
            dtype=torch.float32,
        )
        truncated, indices = _column_top_k_preserve_mass(weights, top_k=1)
        self.assertIsNotNone(indices)
        self.assertEqual(indices.squeeze(0).squeeze(0).tolist(), [0, 1, 2, 2])
        self.assertTrue(torch.equal(truncated.ne(0).sum(dim=1), torch.ones(1, 4)))
        torch.testing.assert_close(truncated.sum(dim=1), weights.sum(dim=1))

    def test_top_p_and_top_k_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            _truncate_force_group(
                torch.rand(2, 3, 5),
                top_p=0.9,
                top_p_min_keep=1,
                top_k=2,
            )

    def _check_indexed_accumulation(self, device: str):
        torch.manual_seed(11)
        batch, n_gen, top_k, c0, c1, features = 2, 4, 3, 5, 6, 9
        weights = torch.randn(batch, n_gen, top_k, device=device)
        indices = torch.randint(c0 + c1, (batch, n_gen, top_k), device=device)
        target0 = torch.randn(batch, c0, features, device=device)
        target1 = torch.randn(batch, c1, features, device=device)

        targets = torch.cat([target0, target1], dim=1)
        selected = torch.gather(
            targets.unsqueeze(1).expand(-1, n_gen, -1, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, features),
        )
        expected = (weights.unsqueeze(-1) * selected).sum(2)
        actual = _accumulate_weighted_targets(
            weights,
            indices,
            target0,
            target1,
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

        previous = torch.randn_like(expected)
        expected_accumulated = previous - 0.7 * expected
        actual_accumulated = _accumulate_weighted_targets(
            weights,
            indices,
            target0,
            target1,
            out=previous.clone(),
            alpha=-0.7,
        )
        torch.testing.assert_close(
            actual_accumulated,
            expected_accumulated,
            atol=2e-5,
            rtol=2e-5,
        )

    def test_indexed_accumulation_cpu(self):
        self._check_indexed_accumulation("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_indexed_accumulation_cuda(self):
        self._check_indexed_accumulation("cuda")

    def test_top_k_at_pool_size_matches_dense_mixed_loss_and_gradient(self):
        torch.manual_seed(17)
        gen = torch.randn(2, 5, 7)
        pos = torch.randn(2, 9, 7)
        neg = torch.randn(2, 3, 7)

        def run(top_k_pos: int, top_k_neg: int):
            gen_run = gen.clone().requires_grad_(True)
            loss, _ = drift_loss_imagenet_mixed(
                gen_run,
                pos,
                neg,
                alpha=0.35,
                R_list_baseline=(0.2, 0.05),
                R_list_versionb=(0.4, 0.1),
                global_scale_stats=False,
                global_fnorm_stats=False,
                top_k_pos=top_k_pos,
                top_k_neg=top_k_neg,
            )
            loss.mean().backward()
            return loss.detach(), gen_run.grad.detach()

        dense_loss, dense_grad = run(0, 0)
        full_loss, full_grad = run(pos.shape[1], gen.shape[1] + neg.shape[1])
        torch.testing.assert_close(full_loss, dense_loss, atol=0.0, rtol=0.0)
        torch.testing.assert_close(full_grad, dense_grad, atol=0.0, rtol=0.0)

    def test_compact_top_k_mixed_loss_has_finite_gradient(self):
        torch.manual_seed(23)
        gen = torch.randn(2, 5, 7, requires_grad=True)
        pos = torch.randn(2, 9, 7)
        neg = torch.randn(2, 3, 7)
        loss, info = drift_loss_imagenet_mixed(
            gen,
            pos,
            neg,
            alpha=0.5,
            R_list_baseline=(0.2,),
            R_list_versionb=(0.4,),
            compute_wpos_stats=True,
            global_scale_stats=False,
            global_fnorm_stats=False,
            top_k_pos=4,
            top_k_neg=3,
        )
        loss.mean().backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(gen.grad).all())
        self.assertEqual(info["top_k/pos_kept"], 4.0)
        self.assertEqual(info["top_k/pos_pool_size"], 5.0)
        self.assertEqual(info["b/top_k/neg_kept"], 3.0)
        self.assertEqual(info["b/top_k/neg_pool_size"], 8.0)


if __name__ == "__main__":
    unittest.main()
