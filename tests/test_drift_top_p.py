import unittest

import torch

from drifting_core.imagenet_loss import (
    _top_p_preserve_mass,
    drift_loss_imagenet,
)


class DriftTopPTest(unittest.TestCase):
    def test_top_p_one_is_exact_original_weight_grid(self):
        torch.manual_seed(7)
        weights = torch.rand(3, 5, 9)

        actual = _top_p_preserve_mass(
            weights, dim=2, top_p=1.0, min_keep=4
        )

        self.assertTrue(torch.equal(actual, weights))

    def test_top_p_truncates_preserves_mass_and_honors_min_keep(self):
        weights = torch.tensor([[[6.0, 3.0, 1.0, 0.0]]])
        actual = _top_p_preserve_mass(
            weights, dim=2, top_p=0.60, min_keep=2
        )

        torch.testing.assert_close(actual.sum(dim=2), weights.sum(dim=2))
        self.assertEqual(int(actual.count_nonzero().item()), 2)
        self.assertEqual(float(actual[..., 2:].abs().sum().item()), 0.0)

    def test_zero_mass_group_remains_zero(self):
        weights = torch.zeros(2, 3, 5)
        actual = _top_p_preserve_mass(
            weights, dim=2, top_p=0.90, min_keep=2
        )

        self.assertTrue(torch.equal(actual, weights))

    def test_groupwise_top_p_keeps_reverse_positive_force_alive(self):
        torch.manual_seed(11)
        gen = torch.zeros(2, 4, 6, requires_grad=True)
        pos = torch.randn(2, 7, 6)
        neg = torch.randn(2, 3, 6)

        loss, info = drift_loss_imagenet(
            gen,
            pos,
            neg,
            R_list=(0.2,),
            compute_wpos_stats=True,
            global_scale_stats=False,
            global_fnorm_stats=False,
            top_p=0.90,
            top_p_min_keep=2,
        )
        loss.sum().backward()

        self.assertGreater(info["loss_0.2"], 0.0)
        self.assertGreaterEqual(info["top_p/pos_kept_mean"], 2.0)
        self.assertGreaterEqual(info["top_p/neg_kept_mean"], 2.0)
        self.assertEqual(info["top_p/pos_zero_row_fraction"], 0.0)
        self.assertTrue(torch.isfinite(gen.grad).all())

    def test_invalid_parameters_are_rejected(self):
        weights = torch.ones(1, 2, 3)
        for top_p in (0.0, -0.1, 1.1):
            with self.subTest(top_p=top_p), self.assertRaises(ValueError):
                _top_p_preserve_mass(weights, dim=2, top_p=top_p)
        with self.assertRaises(ValueError):
            _top_p_preserve_mass(weights, dim=2, top_p=0.95, min_keep=0)


if __name__ == "__main__":
    unittest.main()
