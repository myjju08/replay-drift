import torch

from drifting_core.imagenet_loss import drift_loss_imagenet


def test_half_multiplier_matches_a_duplicated_temperature():
    torch.manual_seed(19)
    gen = torch.randn(2, 4, 5, requires_grad=True)
    pos = torch.randn(2, 3, 5)
    neg = torch.randn(2, 2, 5)

    single, _ = drift_loss_imagenet(
        gen,
        pos,
        neg,
        R_list=(0.2,),
        global_scale_stats=False,
        global_fnorm_stats=False,
        force_multiplier=1.0,
    )
    duplicated, info = drift_loss_imagenet(
        gen,
        pos,
        neg,
        R_list=(0.2, 0.2),
        global_scale_stats=False,
        global_fnorm_stats=False,
        force_multiplier=0.5,
    )

    torch.testing.assert_close(single, duplicated)
    assert info["force_multiplier"] == 0.5
