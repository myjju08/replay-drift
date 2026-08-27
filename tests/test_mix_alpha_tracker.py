import unittest
from itertools import combinations
from unittest import mock

import torch

from train_imagenet_gen import (
    MixAlphaTracker,
    _feature_loss_weights_for_stages,
    _feature_stage_group,
    _sample_stochastic_feature_stages,
    compute_drift_loss_from_features,
)


class MixAlphaTrackerTest(unittest.TestCase):
    @staticmethod
    def _set_counts(
        tracker: MixAlphaTracker,
        *,
        gen_win: float,
        gen_total: float,
        pos_win: float,
        pos_total: float,
    ) -> None:
        tracker.update(
            {
                "raw/gen_winner_count": gen_win,
                "raw/gen_winner_total": gen_total,
                "raw/pos_winner_count": pos_win,
                "raw/pos_winner_total": pos_total,
            }
        )

    def test_hedge_honors_initial_mix_alpha(self) -> None:
        for initial in (0.0, 0.25, 0.5, 1.0):
            with self.subTest(initial=initial):
                tracker = MixAlphaTracker(
                    gamma=0.0,
                    eta=0.0,
                    decay=0.0,
                    mode="hedge_no_time",
                    initial_alpha=initial,
                )
                actual = tracker.compute_mix_alpha(step=0, total_steps=100)
                expected = min(
                    1.0 - tracker._PRIOR_EPS,
                    max(tracker._PRIOR_EPS, initial),
                )
                self.assertAlmostEqual(actual, expected, places=12)

    def test_capacity_normalization_removes_unequal_set_cap(self) -> None:
        tracker = MixAlphaTracker()
        self._set_counts(
            tracker,
            gen_win=64,
            gen_total=64,
            pos_win=64,
            pos_total=128,
        )

        self.assertEqual(tracker.alpha1, 1.0)
        self.assertEqual(tracker.beta1, 0.5)
        self.assertEqual(tracker.alpha1_capacity, 1.0)
        self.assertEqual(tracker.beta1_capacity, 1.0)

    def test_equal_capacity_coverage_has_no_cardinality_bias(self) -> None:
        tracker = MixAlphaTracker(
            gamma=8.0,
            eta=0.1,
            decay=0.0,
            mode="hedge_no_time",
            initial_alpha=0.5,
        )
        self._set_counts(
            tracker,
            gen_win=32,
            gen_total=64,
            pos_win=32,
            pos_total=128,
        )

        self.assertEqual(tracker.alpha1, 0.5)
        self.assertEqual(tracker.beta1, 0.25)
        self.assertEqual(tracker.alpha1_capacity, 0.5)
        self.assertEqual(tracker.beta1_capacity, 0.5)
        self.assertAlmostEqual(
            tracker.compute_mix_alpha(step=100, total_steps=100),
            0.5,
            places=12,
        )

    def test_state_round_trip_preserves_hedge_prior_and_counts(self) -> None:
        source = MixAlphaTracker(initial_alpha=0.0, mode="hedge_no_time")
        self._set_counts(
            source,
            gen_win=12,
            gen_total=64,
            pos_win=12,
            pos_total=128,
        )
        source.compute_mix_alpha(step=10, total_steps=100)

        restored = MixAlphaTracker(initial_alpha=0.5, mode="hedge_no_time")
        restored.load_state_dict(source.state_dict())

        self.assertEqual(restored.state_dict(), source.state_dict())
        self.assertEqual(restored.alpha1_capacity, source.alpha1_capacity)
        self.assertEqual(restored.beta1_capacity, source.beta1_capacity)


class StochasticFeatureStageLossTest(unittest.TestCase):
    def test_feature_keys_are_grouped_by_encoder_stage(self) -> None:
        expected = {
            "global": None,
            "norm_x": None,
            "conv1": "stage1",
            "conv1_std_4": "stage1",
            "layer1_blk2_mean": "stage1",
            "layer2": "stage2",
            "layer3_blk6_std_2": "stage3",
            "layer4_blk2": "stage4",
        }
        for name, group in expected.items():
            with self.subTest(name=name):
                self.assertEqual(_feature_stage_group(name), group)

    def test_two_stage_draw_is_deterministic_and_resume_stable(self) -> None:
        first = _sample_stochastic_feature_stages(
            stage_count=2,
            seed=42,
            step=1234,
        )
        resumed = _sample_stochastic_feature_stages(
            stage_count=2,
            seed=42,
            step=1234,
        )

        self.assertEqual(first, resumed)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(set(first)), 2)

    def test_inverse_probability_weights_recover_full_loss_in_expectation(self) -> None:
        names = ("global", "layer1", "layer2", "layer3", "layer4")
        contributions = {
            "global": 0.5,
            "layer1": 1.0,
            "layer2": 2.0,
            "layer3": 3.0,
            "layer4": 4.0,
        }
        estimates = []
        for selected in combinations(("stage1", "stage2", "stage3", "stage4"), 2):
            weights = _feature_loss_weights_for_stages(names, selected)
            estimates.append(sum(weights[name] * contributions[name] for name in names))

        full_loss = sum(contributions.values())
        self.assertAlmostEqual(sum(estimates) / len(estimates), full_loss)

    def test_zero_weight_skips_feature_and_selected_loss_is_rescaled(self) -> None:
        gen_feats = {
            "global": torch.ones(1, 1, 1),
            "layer1": torch.full((1, 1, 1), 2.0),
            "layer2": torch.full((1, 1, 1), 100.0),
        }
        pos_feats = {name: torch.zeros_like(value) for name, value in gen_feats.items()}
        weights = {"global": 1.0, "layer1": 2.0, "layer2": 0.0}
        calls = []

        def fake_drift_loss(*, gen, **kwargs):
            calls.append(float(gen.mean()))
            return gen.mean().reshape(1), {}

        with mock.patch(
            "train_imagenet_gen.drift_loss_imagenet",
            side_effect=fake_drift_loss,
        ):
            loss, _ = compute_drift_loss_from_features(
                gen_feats=gen_feats,
                pos_feats=pos_feats,
                neg_feats=None,
                B=1,
                G=1,
                P=1,
                N=0,
                weight_neg=None,
                drift_matching="rev-drift",
                feature_loss_weights=weights,
            )

        self.assertEqual(calls, [1.0, 2.0])
        self.assertEqual(loss.item(), 5.0)

    def test_pruning_occurs_after_full_hedge_stats(self) -> None:
        gen_feats = {
            "global": torch.ones(1, 1, 1, requires_grad=True),
            "layer1": torch.full((1, 1, 1), 2.0, requires_grad=True),
            "layer2": torch.full((1, 1, 1), 3.0, requires_grad=True),
            "layer3": torch.full((1, 1, 1), 4.0, requires_grad=True),
            "layer4": torch.full((1, 1, 1), 5.0, requires_grad=True),
        }
        pos_feats = {name: torch.zeros_like(value) for name, value in gen_feats.items()}
        weights = {
            "global": 1.0,
            "layer1": 2.0,
            "layer2": 0.0,
            "layer3": 0.0,
            "layer4": 0.0,
        }

        def fake_drift_loss(*, gen, **kwargs):
            return gen.mean().reshape(1), {}

        with mock.patch(
            "train_imagenet_gen.drift_loss_imagenet",
            side_effect=fake_drift_loss,
        ):
            loss, info = compute_drift_loss_from_features(
                gen_feats=gen_feats,
                pos_feats=pos_feats,
                neg_feats=None,
                B=1,
                G=1,
                P=1,
                N=0,
                weight_neg=None,
                drift_matching="rev-drift",
                compute_raw_winner_stats_flag=True,
                feature_loss_weights=weights,
                prune_zero_weight_features=True,
            )

        # Hedge saw all four base layers before the zero-loss layers vanished.
        self.assertEqual(info["raw/gen_winner_total"], 1.0)
        self.assertEqual(info["raw/pos_winner_total"], 1.0)
        self.assertEqual(info["stochastic_stage/pruned_feature_count"], 3.0)
        self.assertEqual(set(gen_feats), {"global", "layer1"})
        self.assertEqual(set(pos_feats), {"global", "layer1"})
        self.assertEqual(loss.item(), 5.0)


if __name__ == "__main__":
    unittest.main()
