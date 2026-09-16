#!/usr/bin/env python3
"""R1-padding v3(验收二轮修复):
- Transformer 全 PAD 行短路处理:forward+backward 梯度均有限
- 增加 BiLSTM
- full-axis 非 execute 事件用独立 token(不与 PAD=0 冲突)
- seq_module.py 正式序列模块(供 R2 接入旧 GRU/LSTM/residual)
"""
import json
import os
import torch
import torch.nn as nn

D = "/root/autodl-tmp/scheduler/"
OUT = D + "results/processed/behavior_nn_v1/tests/"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42)

PAD = 0
EMB_D = 16
HID = 64


def seq_to_padded(token_seqs, max_len):
    lens = [len(s) for s in token_seqs]
    T = max_len
    toks = [list(s) + [PAD] * (T - len(s)) for s in token_seqs]
    tokens = torch.tensor(toks, dtype=torch.long)
    lengths = torch.tensor(lens, dtype=torch.long)
    mask = torch.zeros(tokens.shape, dtype=torch.bool)
    for i, l in enumerate(lens):
        mask[i, :l] = True
    return tokens, lengths, mask


class PackedRNN(nn.Module):
    """GRU/LSTM/BiLSTM,pad_packed + gather 最后有效位置;len=0 输出 pad 状态(有限)"""

    def __init__(self, cell="gru", bidirectional=False, vocab=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB_D, padding_idx=PAD)
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(EMB_D, HID, batch_first=True, bidirectional=bidirectional)
        d = HID * (2 if bidirectional else 1)
        self.head = nn.Linear(d, 6)

    def forward(self, tokens, lengths):
        e = self.emb(tokens)
        # len=0 行用安全长度 1 参与 pack,输出后掩码清零
        valid = lengths > 0
        l_safe = lengths.clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, l_safe.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        idx = (l_safe - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.shape[2])
        last = out.gather(1, idx).squeeze(1)
        last = torch.where(valid.unsqueeze(1), last, torch.zeros_like(last))
        return self.head(last)


class MaskedTransformer(nn.Module):
    """全 PAD 行短路:不进入 attention(否则 softmax 全 -inf → NaN,反向传播污染)"""

    def __init__(self, vocab=8, d=32, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Parameter(torch.zeros(1, 64, d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.head = nn.Linear(d, 6)

    def _fwd_valid(self, e, mask):
        a, _ = self.attn(e, e, e, key_padding_mask=~mask)
        pooled = (a * mask.unsqueeze(2)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)

    def forward(self, tokens, mask):
        e = self.emb(tokens) + self.pos[:, : tokens.shape[1]]
        valid = mask.any(dim=1)
        if bool(valid.all()):
            return self._fwd_valid(e, mask)
        # invalid 行(含全空 batch):head(零状态) —— 经过可学习 head,设备一致,梯度路径存在
        out = self.head(torch.zeros(tokens.shape[0], e.shape[2], device=tokens.device))
        if bool(valid.any()):
            out[valid] = self._fwd_valid(e[valid], mask[valid])
        return out


def main():
    results = {}
    seqs = [[3, 1, 4, 2, 1], [3, 1, 2], [5]]
    max_len = 8

    # t1 pad invariance(GRU/LSTM/BiLSTM)
    for cell in ("gru", "lstm"):
        for bi in (False, True):
            model = PackedRNN(cell, bidirectional=bi)
            model.eval()
            toks, lens, _ = seq_to_padded(seqs, max_len)
            with torch.no_grad():
                base = model(toks, lens)
            ok = True
            for extra in (3, 5):
                t2, l2, _ = seq_to_padded(seqs, max_len + extra)
                with torch.no_grad():
                    ok = ok and bool(torch.allclose(model(t2, l2), base, atol=1e-5))
            k = f"t1_{cell}{'_bi' if bi else ''}_pad_invariance"
            results[k] = ok
            print(k, ok)

    # t2 真实 len=0 有限(GRU/LSTM)
    for cell in ("gru", "lstm"):
        m = PackedRNN(cell)
        m.eval()
        with torch.no_grad():
            out = m(torch.zeros(2, 4, dtype=torch.long), torch.tensor([0, 0]))
        results[f"t2_{cell}_real_len0"] = bool(torch.isfinite(out).all())
        print(f"t2 {cell} real len=0:", results[f"t2_{cell}_real_len0"])

    # t3 Transformer 全 PAD:forward + backward 梯度有限 + 设备一致 + 全空 batch
    tf = MaskedTransformer()
    mixed_tok, mixed_len, mixed_mask = seq_to_padded([[3, 1], [], [5, 2, 1]], 5)
    out = tf(mixed_tok, mixed_mask)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in tf.parameters() if p.grad is not None]
    all_fin = all(bool(torch.isfinite(g).all()) for g in grads)
    results["t3_transformer_allpad_grad"] = {
        "forward_finite": bool(torch.isfinite(out).all()),
        "gradient_finite": all_fin, "n_params_with_grad": len(grads)}

    # t3b CPU 全空 batch:forward 有限 + backward 梯度存在且有限
    tf2 = MaskedTransformer()
    empty_tok = torch.zeros(4, 6, dtype=torch.long)
    empty_mask = torch.zeros(4, 6, dtype=torch.bool)
    out_e = tf2(empty_tok, empty_mask)
    loss_e = out_e.sum()
    loss_e.backward()
    g2 = [p.grad for p in tf2.parameters() if p.grad is not None]
    results["t3b_cpu_all_empty"] = {
        "forward_finite": bool(torch.isfinite(out_e).all()),
        "gradients_exist": len(g2) > 0,
        "gradient_finite": all(bool(torch.isfinite(g).all()) for g in g2)}
    print("t3b CPU all-empty batch:", results["t3b_cpu_all_empty"])

    # t3c CUDA 混合 batch + 全空(可用时)
    if torch.cuda.is_available():
        tf3 = MaskedTransformer().cuda()
        mt = mixed_tok.cuda(); mm = mixed_mask.cuda()
        o = tf3(mt, mm)
        o.sum().backward()
        g3 = [p.grad for p in tf3.parameters() if p.grad is not None]
        tf4 = MaskedTransformer().cuda()
        et = torch.zeros(4, 6, dtype=torch.long, device="cuda")
        em = torch.zeros(4, 6, dtype=torch.bool, device="cuda")
        oe = tf4(et, em)
        oe.sum().backward()
        g4 = [p.grad for p in tf4.parameters() if p.grad is not None]
        results["t3c_cuda"] = {
            "mixed_grad_finite": all(bool(torch.isfinite(g).all()) for g in g3),
            "all_empty_grad_finite": all(bool(torch.isfinite(g).all()) for g in g4),
            "all_empty_grad_exist": len(g4) > 0}
        print("t3c CUDA:", results["t3c_cuda"])
    else:
        results["t3c_cuda"] = "no_cuda_skipped"
        print("t3c CUDA: skipped(no cuda)")

    # t4 batch vs single(GRU/LSTM/BiLSTM)
    for cell in ("gru", "lstm"):
        for bi in (False, True):
            m = PackedRNN(cell, bidirectional=bi)
            m.eval()
            toks, lens, _ = seq_to_padded(seqs, max_len)
            with torch.no_grad():
                b = m(toks, lens)
                s = torch.stack([m(t[None], l[None]) for t, l in zip(toks, lens)]).squeeze(1)
            results[f"t4_{cell}{'_bi' if bi else ''}_batch_vs_single"] = bool(torch.allclose(b, s, atol=1e-5))

    # t5 mask vs lengths
    toks, lens, mask = seq_to_padded(seqs, max_len)
    results["t5_mask_lengths"] = int(mask.sum()) == int(lens.sum())

    # t6 P99+1(canonical 数据)
    samples = [json.loads(l) for l in open(OUT.replace("tests/", "data/") + "semantic_tool_samples.jsonl")]
    lens_real = sorted(s["target_event_index"] for s in samples)
    p99 = lens_real[min(len(lens_real) - 1, int(0.99 * len(lens_real)))]
    cap = min(p99 + 1, 64)
    m = PackedRNN("gru"); m.eval()
    with torch.no_grad():
        long_seq = [1] * cap
        t6, l6, _ = seq_to_padded([long_seq], cap)
        results["t6_p99plus1"] = {"p99": p99, "cap": cap,
                                  "finite": bool(torch.isfinite(m(t6, l6)).all())}
    print("t6 P99+1 cap:", cap)

    # t7 canonical smoke:非 execute 用独立 token(1),execute 族 2-7,PAD=0
    fam_idx = {"select_frames": 2, "visual_qa": 3, "temporal_ops": 4, "summarize": 5, "detect": 6, "other": 7}
    FAM = {"sample_seek": "select_frames", "frame-selector": "select_frames",
           "image-grid-selector": "select_frames", "spatial_qa": "visual_qa",
           "image-qa": "visual_qa", "image-grid-qa": "visual_qa", "patch-zoomer": "visual_qa",
           "temporal-qa": "temporal_ops", "temporal-grounding": "temporal_ops",
           "summarize": "summarize", "summarization-tool": "summarize",
           "object_detection": "detect", "yolo-tracker": "detect"}
    rows_all = [json.loads(l) for l in open(D + "results/processed/role_dataset_v0_3.jsonl")]
    from collections import defaultdict
    by_run = defaultdict(list)
    for r in rows_all:
        by_run[r["run_id"]].append(r)
    for rl in by_run.values():
        rl.sort(key=lambda x: x["event_index"])

    def tok_of(r):
        if r["role"] == "execute":
            return fam_idx.get(FAM.get(r["raw_action"], "other"), 7)
        return 1  # 非 execute 事件(plan/aggregate/terminate)独立 token

    seqs_full, seqs_exe = [], []
    for s in samples[:200]:
        rl = [r for r in by_run.get(s["run_id"], []) if r["event_index"] < s["target_event_index"]]
        seqs_full.append([tok_of(r) for r in rl][-64:])
        seqs_exe.append([tok_of(r) for r in rl if r["role"] == "execute"][-64:])
    m = PackedRNN("gru"); m.eval()
    with torch.no_grad():
        t7, l7, _ = seq_to_padded(seqs_full, 64)
        ok_full = bool(torch.isfinite(m(t7, l7)).all())
        t7e, l7e, _ = seq_to_padded(seqs_exe, 64)
        ok_exe = bool(torch.isfinite(m(t7e, l7e)).all())
    non_exe_kept = sum(1 for s in seqs_full if 1 in s)
    results["t7_canonical_smoke"] = {"full_axis": ok_full, "execute_only_axis": ok_exe,
                                     "n": len(seqs_full), "non_execute_tokens_kept_samples": non_exe_kept,
                                     "max_full_len": max(map(len, seqs_full))}
    print("t7 canonical smoke:", results["t7_canonical_smoke"])

    with open(OUT + "padding_report.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("written:", OUT + "padding_report.json")


if __name__ == "__main__":
    main()
