import copy
import unittest

import torch

from models.feature_adapter import (
    FeatureAdapterSystem,
    ResidualSpatialAdapter,
    canonical_adapter_stages,
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


if __name__ == "__main__":
    unittest.main()
