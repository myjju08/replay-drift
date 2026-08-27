import torch

from drifting_core.topk_diagnostics import diagnose_reverse_topk_heterogeneity


def test_identical_features_have_identical_topk_supports():
    torch.manual_seed(7)
    B, G, P, N, T, D = 2, 4, 4, 2, 3, 5
    gen = torch.randn(B * G, T, D)
    pos = torch.randn(B * P, T, D)
    neg = torch.randn(B * N, T, D)
    metrics = diagnose_reverse_topk_heterogeneity(
        gen_feats={"layer1": gen, "layer2": gen.clone()},
        pos_feats={"layer1": pos, "layer2": pos.clone()},
        neg_feats={"layer1": neg, "layer2": neg.clone()},
        batch_size=B,
        gen_count=G,
        pos_count=P,
        neg_count=N,
        weight_neg=torch.ones(B, N),
        R_list=(0.2, 0.05),
        top_k_pos=2,
        top_k_neg=3,
        global_scale_stats=False,
    )

    assert metrics["topk_diag/pos/consensus_stage1_to_stage2_overlap"] == 1.0
    assert metrics["topk_diag/neg/consensus_stage1_to_stage2_overlap"] == 1.0
    assert metrics["topk_diag/pos/layer1_to_stage2_overlap"] == 1.0
    assert metrics["topk_diag/neg/layer1_to_stage2_overlap"] == 1.0


def test_candidate_union_metrics_are_bounded():
    torch.manual_seed(11)
    B, G, P, N, T, D = 1, 3, 5, 2, 2, 4
    metrics = diagnose_reverse_topk_heterogeneity(
        gen_feats={"layer1": torch.randn(B * G, T, D)},
        pos_feats={"layer1": torch.randn(B * P, T, D)},
        neg_feats={"layer1": torch.randn(B * N, T, D)},
        batch_size=B,
        gen_count=G,
        pos_count=P,
        neg_count=N,
        weight_neg=torch.ones(B, N),
        R_list=(0.2,),
        top_k_pos=2,
        top_k_neg=2,
        global_scale_stats=False,
    )

    for key, value in metrics.items():
        if "overlap" in key or "union" in key:
            assert 0.0 <= value <= 1.0, (key, value)

    assert (
        metrics["topk_diag/pos/token_query_union_across_features_stage1"]
        >= metrics["topk_diag/pos/token_query_union_stage1"]
    )
