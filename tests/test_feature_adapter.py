import copy
import unittest

import torch

from memory_bank import ArrayMemoryBank
from models.feature_adapter import (
    FeatureAdapterSystem,
    ResidualSpatialAdapter,
    canonical_adapter_stages,
    multi_positive_info_nce_loss,
    supervised_contrastive_loss,
    update_adapter_ema,
)
from models.mae_resnet import MAEResNet


class FeatureAdapterTest(unittest.TestCase):
    def test_canonical_adapter_stages_accepts_layer_and_stage_names(self):
        self.assertEqual(
            canonical_adapter_stages(["layer4", "stage3", "layer4"]),
            ("stage3", "stage4"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown feature-adapter key"):
            canonical_adapter_stages(["layer5"])

    def test_residual_adapter_is_exact_identity_at_initialization(self):
        adapter = ResidualSpatialAdapter(channels=8, bottleneck=3)
        x = torch.randn(4, 8, 5, 5)
        torch.testing.assert_close(adapter(x), x, rtol=0.0, atol=0.0)

    def test_supcon_adapter_objective_is_finite_and_updates_online_only(self):
        system = FeatureAdapterSystem(
            {"stage3": 8, "stage4": 16},
            ["layer3", "layer4"],
            bottleneck=4,
            projection_dim=6,
            num_classes=5,
            use_ce=True,
        )
        target = copy.deepcopy(system).eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        features = {
            "layer3": torch.randn(6, 8, 4, 4),
            "layer4": torch.randn(6, 16, 2, 2),
        }
        labels = torch.tensor([1, 2])
        target_before = [parameter.clone() for parameter in target.parameters()]
        loss, metrics = system(
            features,
            labels,
            batch_size=2,
            positive_count=3,
            samples_per_class=2,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.1,
            reg_weight=0.01,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("adapter/stage3_supcon", metrics)
        loss.backward()
        self.assertIsNotNone(system.adapters["stage4"].up.weight.grad)
        self.assertTrue(torch.isfinite(system.adapters["stage4"].up.weight.grad).all())
        torch.optim.SGD(system.parameters(), lr=0.1).step()
        update_adapter_ema(target, system, decay=0.9)
        self.assertTrue(all(not parameter.requires_grad for parameter in target.parameters()))
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(target_before, target.parameters())
            )
        )

    def test_supervised_contrastive_rejects_singletons(self):
        with self.assertRaisesRegex(ValueError, "same-label positive"):
            supervised_contrastive_loss(
                torch.randn(3, 4), torch.tensor([0, 1, 2]), temperature=0.1
            )

    def test_asymmetric_multi_positive_infonce_matches_explicit_formula(self):
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        anchor_labels = torch.tensor([0, 1])
        candidates = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
        )
        candidate_labels = torch.tensor([0, 0, 1, 1])
        temperature = 0.2
        loss = multi_positive_info_nce_loss(
            anchors,
            anchor_labels,
            candidates,
            candidate_labels,
            temperature,
        )

        logits = torch.nn.functional.normalize(anchors, dim=-1) @ (
            torch.nn.functional.normalize(candidates, dim=-1).t()
        )
        logits = logits / temperature
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive = anchor_labels[:, None].eq(candidate_labels[None, :])
        expected = -(
            log_prob.masked_fill(~positive, 0.0).sum(dim=1)
            / positive.sum(dim=1)
        ).mean()
        torch.testing.assert_close(loss, expected)

    def test_generated_real_infonce_detaches_inputs_and_updates_adapter_only(self):
        system = FeatureAdapterSystem(
            {"stage3": 8},
            ["stage3"],
            bottleneck=4,
            projection_dim=6,
            num_classes=5,
            objective="gen_real_multipos_infonce",
        )
        self.assertIsInstance(system.projectors["stage3"], torch.nn.Identity)
        real_features = {
            "layer3": torch.randn(6, 8, 4, 4, requires_grad=True)
        }
        generated_features = {
            "layer3": torch.randn(4, 8, 4, 4, requires_grad=True)
        }
        labels = torch.tensor([1, 2])
        loss, metrics = system(
            real_features,
            labels,
            batch_size=2,
            positive_count=3,
            samples_per_class=3,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.0,
            reg_weight=0.0,
            generated_stage_features=generated_features,
            generated_count=2,
            generated_samples_per_class=2,
            generated_anchor_weight=1.0,
            real_anchor_weight=0.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("adapter/stage3_gen_to_real_infonce", metrics)
        self.assertIn("adapter/stage3_real_to_real_infonce", metrics)
        self.assertEqual(metrics["adapter/stage3_real_to_real_infonce"].item(), 0.0)
        self.assertIn("adapter/stage3_positive_cosine", metrics)
        self.assertIn("adapter/stage3_negative_cosine", metrics)
        loss.backward()
        self.assertIsNone(real_features["layer3"].grad)
        self.assertIsNone(generated_features["layer3"].grad)
        self.assertIsNotNone(system.adapters["stage3"].up.weight.grad)
        self.assertTrue(
            torch.isfinite(system.adapters["stage3"].up.weight.grad).all()
        )

    def test_alternating_step_preserves_generator_grad_and_hard_copies_target(self):
        online = FeatureAdapterSystem(
            {"stage3": 8},
            ["stage3"],
            bottleneck=4,
            projection_dim=6,
            objective="gen_real_multipos_infonce",
        )
        target = copy.deepcopy(online).eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)

        generated = torch.randn(4, 8, 4, 4, requires_grad=True)
        # Surrogate for the generator drift path: only the frozen target
        # adapter is used during this backward.
        target.adapters["stage3"](generated).square().mean().backward()
        generator_grad_before = generated.grad.detach().clone()

        real = torch.randn(6, 8, 4, 4)
        adapter_loss, _ = online(
            {"layer3": real},
            torch.tensor([1, 2]),
            batch_size=2,
            positive_count=3,
            samples_per_class=3,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.0,
            reg_weight=0.0,
            generated_stage_features={"layer3": generated},
            generated_count=2,
            generated_samples_per_class=2,
            generated_anchor_weight=1.0,
            real_anchor_weight=0.0,
        )
        adapter_loss.backward()
        torch.testing.assert_close(generated.grad, generator_grad_before)
        self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))

        torch.optim.SGD(online.parameters(), lr=0.1).step()
        update_adapter_ema(target, online, decay=0.0)
        for target_parameter, online_parameter in zip(
            target.parameters(), online.parameters()
        ):
            torch.testing.assert_close(target_parameter, online_parameter)

    def test_s4_mask_has_32_positives_and_128_real_candidates(self):
        system = FeatureAdapterSystem(
            {"stage3": 8},
            ["stage3"],
            bottleneck=4,
            objective="gen_real_multipos_infonce",
        )
        loss, metrics = system(
            {"layer3": torch.randn(4 * 32, 8, 1, 1)},
            torch.tensor([10, 20, 30, 40]),
            batch_size=4,
            positive_count=32,
            samples_per_class=32,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.0,
            reg_weight=0.0,
            generated_stage_features={
                "layer3": torch.randn(4 * 32, 8, 1, 1)
            },
            generated_count=32,
            generated_samples_per_class=32,
            generated_anchor_weight=1.0,
            real_anchor_weight=0.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            metrics["adapter/stage3_positives_per_anchor"].item(), 32.0
        )
        self.assertEqual(
            metrics["adapter/stage3_real_candidate_count"].item(), 128.0
        )

    def test_empty_negative_metric_is_finite_and_distinct_bank_sampling_is_unique(self):
        system = FeatureAdapterSystem(
            {"stage3": 8},
            ["stage3"],
            bottleneck=4,
            objective="gen_real_multipos_infonce",
        )
        loss, metrics = system(
            {"layer3": torch.randn(32, 8, 1, 1)},
            torch.tensor([7]),
            batch_size=1,
            positive_count=32,
            samples_per_class=32,
            temperature=0.1,
            supcon_weight=1.0,
            ce_weight=0.0,
            reg_weight=0.0,
            generated_stage_features={"layer3": torch.randn(32, 8, 1, 1)},
            generated_count=32,
            generated_samples_per_class=32,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["adapter/stage3_negative_cosine"].item(), 0.0)

        bank = ArrayMemoryBank(num_classes=1, max_size=32)
        bank.add(torch.arange(32, dtype=torch.float32).unsqueeze(1), torch.zeros(32))
        self.assertTrue(bank.is_ready(32))
        sampled = bank.sample(torch.tensor([0]), n_samples=32)
        self.assertEqual(torch.unique(sampled).numel(), 32)

    def test_mae_stage_adapter_hook_preserves_frozen_control_and_identity(self):
        mae = MAEResNet(
            num_classes=10,
            in_channels=4,
            base_channels=8,
            layers=(1, 1, 1, 1),
            use_bf16=False,
            input_patch_size=1,
        ).eval()
        x = torch.randn(2, 4, 8, 8)
        kwargs = dict(
            patch_mean_size=[],
            patch_std_size=[],
            use_mean=True,
            use_std=True,
            with_global=True,
            every_k_block=float("inf"),
        )
        baseline = mae.get_activations(x, **kwargs)
        system = FeatureAdapterSystem(
            {"stage4": 64}, ["stage4"], bottleneck=4, projection_dim=8
        )
        adapted, stage_features = mae.get_activations(
            x,
            **kwargs,
            stage_adapters=system.adapters,
            return_stage_features=True,
        )
        self.assertEqual(
            set(stage_features), {"layer1", "layer2", "layer3", "layer4"}
        )
        self.assertEqual(baseline.keys(), adapted.keys())
        for key in baseline:
            torch.testing.assert_close(
                baseline[key], adapted[key], rtol=0.0, atol=0.0
            )

    def test_mae_active_stages_omit_early_feature_computation(self):
        mae = MAEResNet(
            num_classes=10,
            in_channels=4,
            base_channels=8,
            layers=(2, 2, 2, 2),
            use_bf16=False,
            input_patch_size=1,
        ).eval()
        activations = mae.get_activations(
            torch.randn(2, 4, 8, 8),
            patch_mean_size=[2],
            patch_std_size=[2],
            use_mean=True,
            use_std=True,
            with_global=True,
            every_k_block=1,
            active_stages=["stage3", "stage4"],
        )

        self.assertIn("global", activations)
        self.assertIn("norm_x", activations)
        self.assertTrue(any(name.startswith("layer3") for name in activations))
        self.assertTrue(any(name.startswith("layer4") for name in activations))
        self.assertFalse(
            any(
                name == "conv1"
                or name.startswith("conv1_")
                or name == "layer1"
                or name.startswith("layer1_")
                or name == "layer2"
                or name.startswith("layer2_")
                for name in activations
            )
        )

        _, stage_features = mae.get_activations(
            torch.randn(2, 4, 8, 8),
            patch_mean_size=[2],
            patch_std_size=[2],
            use_mean=True,
            use_std=True,
            with_global=True,
            every_k_block=1,
            active_stages=["stage3", "stage4"],
            return_stage_features=True,
        )
        self.assertEqual(set(stage_features), {"layer3", "layer4"})

        with self.assertRaisesRegex(ValueError, "Unknown active feature stages"):
            mae.get_activations(
                torch.randn(1, 4, 8, 8), active_stages=["stage5"]
            )


if __name__ == "__main__":
    unittest.main()
