import unittest

import torch

from models.feature_gan import (
    FrozenFeatureDiscriminator,
    canonical_feature_gan_stages,
    discriminator_hinge_loss,
    generator_hinge_loss,
)


class FeatureGANTest(unittest.TestCase):
    def test_canonical_stages_accept_layer_and_stage_names(self):
        self.assertEqual(
            canonical_feature_gan_stages(["layer4", "stage3", "layer4"]),
            ("stage3", "stage4"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown feature-GAN stage"):
            canonical_feature_gan_stages(["layer5"])

    def test_multiscale_conditional_discriminator_backpropagates_to_features(self):
        discriminator = FrozenFeatureDiscriminator(
            {"stage3": 8, "stage4": 16},
            ["layer3", "layer4"],
            hidden_channels=6,
            num_classes=5,
        )
        features = {
            "stage3": torch.randn(4, 8, 4, 4, requires_grad=True),
            "stage4": torch.randn(4, 16, 2, 2, requires_grad=True),
        }
        labels = torch.tensor([0, 1, 2, 3])
        scores = discriminator(features, labels)
        self.assertEqual(scores.shape, (4,))
        generator_hinge_loss(scores).backward()
        for feature in features.values():
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())

    def test_hinge_losses_match_definition(self):
        real = torch.tensor([2.0, 0.0])
        fake = torch.tensor([-2.0, 0.0])
        self.assertAlmostEqual(discriminator_hinge_loss(real, fake).item(), 1.0)
        self.assertAlmostEqual(generator_hinge_loss(fake).item(), 1.0)

    def test_gradient_calibration_is_resumable_buffer_state(self):
        discriminator = FrozenFeatureDiscriminator(
            {"stage4": 8}, ["stage4"], hidden_channels=4, num_classes=3
        )
        discriminator.update_gradient_calibration(6.0, 2.0, ema_decay=0.9)
        self.assertAlmostEqual(discriminator.gradient_unit_scale.item(), 3.0)
        self.assertEqual(discriminator.gradient_ratio_updates.item(), 1)

        restored = FrozenFeatureDiscriminator(
            {"stage4": 8}, ["stage4"], hidden_channels=4, num_classes=3
        )
        restored.load_state_dict(discriminator.state_dict())
        self.assertAlmostEqual(restored.gradient_unit_scale.item(), 3.0)
        self.assertEqual(restored.gradient_ratio_updates.item(), 1)


if __name__ == "__main__":
    unittest.main()
