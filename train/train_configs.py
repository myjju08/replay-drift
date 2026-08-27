"""Default training configurations for ImageNet drifting models."""

# ---------------------------------------------------------------------------
# MAE pretraining
# ---------------------------------------------------------------------------

IMAGENET_MAE_PIXEL_CONFIG = {
    "resolution": 256,
    "use_aug": True,
    "use_latent": False,
    "use_cache": False,
    "num_classes": 1000,
    "batch_size": 512,
    "eval_batch_size": 512,
    "num_workers": 8,
    "pin_memory": True,
    # model
    "base_channels": 640,
    "patch_size": 2,
    "dropout_prob": 0.0,
    "layers": [3, 4, 6, 3],
    "in_channels": 3,
    "use_bf16": True,
    "input_patch_size": 8,        # 256 → 32 before encoder
    # optimizer
    "lr": 4e-3,
    "weight_decay": 0.01,
    "adam_b1": 0.9,
    "adam_b2": 0.95,
    "warmup_steps": 4000,
    "lr_schedule": "const",
    # training
    "seed": 42,
    "total_steps": 200000,
    "save_per_step": 5000,
    "eval_per_step": 2000,
    "eval_samples": 5000,
    "ema_decay": 0.9995,
    "max_grad_norm": 2.0,
    "keep_every": 50000,
    "keep_last": 2,
    "mask_ratio_min": 0.5,
    "mask_ratio_max": 0.5,
    "lambda_cls": 0.0,
    "finetune_last_steps": 3000,
    "warmup_finetune": 1000,
    "finetune_cls": 0.1,
}

IMAGENET_MAE_LATENT_CONFIG = {
    **IMAGENET_MAE_PIXEL_CONFIG,
    "use_latent": True,
    "use_cache": False,
    "in_channels": 4,             # VAE latent channels
    "input_patch_size": 1,        # 32×32 latent — no spatial downsampling
    "batch_size": 512,
}

# ---------------------------------------------------------------------------
# Generator training (DitGen-B)
# ---------------------------------------------------------------------------

IMAGENET_GEN_LATENT_B_CONFIG = {
    "resolution": 256,
    "use_aug": False,
    "use_latent": True,
    "use_cache": True,
    "num_classes": 1000,
    "batch_size": 128,
    "eval_batch_size": 256,
    "num_workers": 8,
    "pin_memory": True,
    # model (DitGen-B)
    "cond_dim": 768,
    "input_size": 32,
    "in_channels": 4,
    "patch_size": 2,
    "hidden_size": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "out_channels": 4,
    "use_qk_norm": True,
    "use_swiglu": True,
    "use_rope": True,
    "use_rmsnorm": True,
    "n_cls_tokens": 16,
    "noise_classes": 64,
    "noise_coords": 32,
    "use_bf16": True,
    # optimizer
    "lr": 4e-4,
    "weight_decay": 0.0,
    "adam_b1": 0.9,
    "adam_b2": 0.95,
    "warmup_steps": 10000,
    "lr_schedule": "const",
    # training loop
    "seed": 42,
    "total_steps": 200000,
    "save_per_step": 2000,
    "eval_per_step": 5000,
    "eval_samples": 50000,
    "ema_decay": 0.999,
    "max_grad_norm": 2.0,
    "keep_every": 50000,
    "keep_last": 2,
    # memory bank
    "pos_per_sample": 64,
    "neg_per_sample": 32,
    "positive_bank_size": 128,
    "negative_bank_size": 1000,
    "push_per_step": 128,
    "push_at_resume": 3000,
    # generator forward
    "gen_per_label": 64,
    "cfg_min": 1.0,
    "cfg_max": 4.0,
    "neg_cfg_pw": 5.0,
    "no_cfg_frac": 0.0,
    # drift loss
    "R_list": [0.2, 0.05, 0.02],
    # activation kwargs
    "activation_kwargs": {
        "patch_mean_size": [2, 4],
        "patch_std_size": [2, 4],
        "use_std": True,
        "use_mean": True,
        "with_global": True,
        "every_k_block": 2,
    },
    # eval CFG scales
    "cfg_list": [1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5],
    # MAE checkpoint (fill in before training)
    "mae_checkpoint": "",
}

IMAGENET_GEN_PIXEL_B_CONFIG = {
    **IMAGENET_GEN_LATENT_B_CONFIG,
    "use_latent": False,
    "use_cache": False,
    "use_aug": True,
    "in_channels": 3,
    "input_size": 256,
    "patch_size": 16,
    "out_channels": 3,
    "batch_size": 64,
    "lr": 2e-4,
    "weight_decay": 0.01,
    "total_steps": 100000,
}
