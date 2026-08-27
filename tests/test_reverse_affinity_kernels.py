import unittest

import torch
import torch.nn.functional as F

from drifting_core.imagenet_loss import (
    _adaptive_reverse_bandwidth,
    _reverse_kernel_weights,
    _reverse_mutual_affinity,
    drift_loss_imagenet,
)


class ReverseAffinityKernelTest(unittest.TestCase):
    def test_exponential_helper_preserves_original_formula(self):
        torch.manual_seed(31)
        distances = torch.rand(2, 4, 9)
        self_mask = torch.zeros(1, 4, 9)
        expected_logits = -distances / 0.05
        expected = (
            F.softmax(expected_logits, dim=2)
            * F.softmax(expected_logits, dim=1)
        ).clamp(min=1e-6).sqrt()
        actual = _reverse_mutual_affinity(
            distances,
            bandwidth=0.05,
            kernel="exponential",
            shape=1.0,
            local_bandwidth=None,
            self_mask=self_mask,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_adaptive_bandwidth_covers_both_target_groups(self):
        distances = torch.tensor(
            [[[0.0, 0.2, 0.4, 0.1, 0.3, 0.5],
              [0.2, 0.0, 0.5, 0.15, 0.35, 0.55],
              [0.4, 0.5, 0.0, 0.12, 0.32, 0.52]]]
        )
        self_mask = F.pad(torch.eye(3), (0, 3)).unsqueeze(0)
        bandwidth = _adaptive_reverse_bandwidth(
            distances,
            split_idx=3,
            self_mask=self_mask,
            k_pos=2,
            k_neg=1,
            margin=1.05,
        )
        self.assertEqual(tuple(bandwidth.shape), (1, 3, 1))
        self.assertTrue(torch.isfinite(bandwidth).all())
        for row in range(3):
            radius = bandwidth[0, row, 0]
            pos_count = (distances[0, row, 3:] < radius).sum()
            neg_dist = distances[0, row, :3].clone()
            neg_dist[row] = float("inf")
            neg_count = (neg_dist < radius).sum()
            self.assertGreaterEqual(int(pos_count), 2)
            self.assertGreaterEqual(int(neg_count), 1)

    def test_non_exponential_losses_are_finite_and_differentiable(self):
        for kernel, shape in (
            ("exponential", 1.0),
            ("generalized_exponential", 0.75),
            ("generalized_exponential", 2.0),
            ("matern_32", 1.0),
            ("tapered_exponential", 1.0),
            ("power_law", 1.0),
            ("student_t", 1.0),
            ("wendland", 1.0),
            ("cauchy_wendland", 1.0),
        ):
            with self.subTest(kernel=kernel):
                torch.manual_seed(37)
                gen = torch.randn(2, 5, 7, requires_grad=True)
                pos = torch.randn(2, 8, 7)
                neg = torch.randn(2, 4, 7)
                loss, info = drift_loss_imagenet(
                    gen,
                    pos,
                    neg,
                    R_list=(1.0,),
                    global_scale_stats=False,
                    global_fnorm_stats=False,
                    affinity_kernel=kernel,
                    kernel_shape=shape,
                    kernel_adaptive_k_pos=3,
                    kernel_adaptive_k_neg=2,
                    kernel_adaptive_margin=1.05,
                    force_multiplier=3.0,
                )
                loss.sum().backward()
                self.assertTrue(torch.isfinite(loss).all())
                self.assertTrue(torch.isfinite(gen.grad).all())
                self.assertGreater(info["kernel_bandwidth_mean"], 0.0)
                self.assertEqual(info["force_multiplier"], 3.0)

    def test_new_kernel_formulas(self):
        scaled = torch.tensor([0.0, 0.5, 1.0, 2.0])
        genexp = _reverse_kernel_weights(
            scaled, kernel="generalized_exponential", shape=0.75
        )
        self.assertTrue(torch.allclose(genexp, torch.exp(-scaled.pow(0.75))))

        z = scaled * (3.0 ** 0.5)
        matern = _reverse_kernel_weights(scaled, kernel="matern_32", shape=1.0)
        self.assertTrue(torch.allclose(matern, (1.0 + z) * torch.exp(-z)))

        tapered = _reverse_kernel_weights(
            scaled, kernel="tapered_exponential", shape=1.0
        )
        expected = torch.exp(-scaled) * (1.0 - scaled).clamp_min(0.0).square()
        self.assertTrue(torch.equal(tapered, expected))
        self.assertTrue(torch.equal(tapered[2:], torch.zeros(2)))

    def test_cauchy_wendland_mix_endpoints_match_components(self):
        torch.manual_seed(41)
        distances = torch.rand(2, 4, 9)
        self_mask = F.pad(torch.eye(4), (0, 5)).unsqueeze(0)
        local_bandwidth = torch.rand(2, 4, 1).add(0.5)
        common = {
            "distances": distances,
            "bandwidth": 1.0,
            "shape": 1.0,
            "local_bandwidth": local_bandwidth,
            "self_mask": self_mask,
        }
        cauchy = _reverse_mutual_affinity(kernel="student_t", **common)
        wendland = _reverse_mutual_affinity(kernel="wendland", **common)
        mix_cauchy = _reverse_mutual_affinity(
            kernel="cauchy_wendland", mix_weight=1.0, **common
        )
        mix_wendland = _reverse_mutual_affinity(
            kernel="cauchy_wendland", mix_weight=0.0, **common
        )
        self.assertTrue(torch.allclose(mix_cauchy, cauchy))
        self.assertTrue(torch.allclose(mix_wendland, wendland))

    def test_invalid_mix_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mix weight"):
            drift_loss_imagenet(
                torch.randn(1, 3, 5),
                torch.randn(1, 4, 5),
                R_list=(1.0,),
                affinity_kernel="cauchy_wendland",
                kernel_adaptive_k_pos=2,
                kernel_adaptive_k_neg=1,
                kernel_mix_weight=1.1,
            )

    def test_kernel_temperature_mixture_is_combined_before_normalization(self):
        distances = torch.tensor([[[0.0, 0.4, 0.8], [0.4, 0.0, 0.6]]])
        self_mask = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        local_bandwidth = torch.tensor([[[1.2], [0.9]]])
        actual = _reverse_mutual_affinity(
            distances,
            bandwidth=1.0,
            kernel="generalized_exponential",
            shape=2.0,
            local_bandwidth=local_bandwidth,
            self_mask=self_mask,
            temperature_mix=(0.45, 0.75),
            temperature_mix_weights=(0.5, 0.5),
        )
        base = distances / local_bandwidth
        weights = 0.5 * torch.exp(-(base / 0.45).square())
        weights += 0.5 * torch.exp(-(base / 0.75).square())
        weights = weights.masked_fill(self_mask.bool(), 0.0)
        expected = (
            weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-12)
            * weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        ).sqrt()
        self.assertTrue(torch.allclose(actual, expected))

    def test_invalid_kernel_temperature_mixture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must match temperatures"):
            _reverse_mutual_affinity(
                torch.ones(1, 2, 3),
                bandwidth=1.0,
                kernel="generalized_exponential",
                shape=2.0,
                local_bandwidth=torch.ones(1, 2, 1),
                self_mask=torch.zeros(1, 2, 3),
                temperature_mix=(0.45, 0.75),
                temperature_mix_weights=(1.0,),
            )


if __name__ == "__main__":
    unittest.main()
