"""
Banco de avaliação do transcritor contra o ground-truth da faixa de demonstração.

Métrica adequada a transcription **multi-instrumento**: um golpe de bumbo e um de chimel
no mesmo pulso são dois acertos, não um erro de rótulo. Portanto o pareamento é feito
por (tempo, pista), com atribuição gulosa e tolerância temporal padrão de 40 ms
(≈ 1/4 de semicolcheia a 100 BPM).

  * onset_recall / precision ..... tempo ± tolerância, qualquer pista
  * hit_recall / hit_precision ... tempo ± tolerância E pista correta
  * pos_accuracy .................. dos hits corretos, fração com |Δtick| ≤ 1
  * velocity ........................ Pearson e MAE (MIDI) nos hits corretos
  * lane_confusion .................. GT detectado, porém rotulado em outra pista
  * bpm_err_pct / meter_ok
Uso: python3 tests/evaluate.py [--sens 1.0] [--file samples/demo_drums.wav] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drumscribe.pipeline import transcribe_bytes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(score_dict: dict, dets: list, gt: dict, tol_s: float = 0.040,
             tick_tol: int = 1) -> dict:
    ev = gt["events"]
    g_t = np.array([e["time"] for e in ev])
    g_lane = np.array([e["lane"] for e in ev])
    g_bar = np.array([e["bar"] for e in ev])
    g_tick = np.array([e["tick"] for e in ev])
    g_vel = np.array([e["velocity"] for e in ev])
    d_t = np.array([d["time"] for d in dets])
    d_lane = np.array([d["lane"] for d in dets])
    d_bar = np.array([d["bar"] for d in dets])
    d_tick = np.array([d["tick"] for d in dets])
    d_vel = np.array([d["velocity"] for d in dets])
    # alinhamento de compasso por *moda* dos pares: um FP na introdução não pode
    # deslocar a métrica inteira (min() era frágil justamente a isso)
    g_off, d_off = 0, 0

    out = {"n_gt": int(g_t.size), "n_notated": int(d_t.size)}
    if g_t.size == 0 or d_t.size == 0:
        out.update({"onset_recall": 0.0, "hit_recall": 0.0, "hit_precision": 0.0})
        return out

    used = np.zeros(d_t.size, dtype=bool)
    tp = np.zeros(g_t.size, dtype=bool)
    matched_det = np.full(g_t.size, -1, dtype=np.int64)
    order = np.argsort(-g_vel)
    for i in order:                                   # guloso: mais forte primeiro
        cand = np.flatnonzero((~used) & (np.abs(d_t - g_t[i]) <= tol_s) & (d_lane == g_lane[i]))
        if cand.size == 0:
            continue
        j = int(cand[np.argmin(np.abs(d_tick[cand] - g_tick[i]))])
        used[j] = True
        tp[i] = True
        matched_det[i] = j
    # recall/precisão de tempo (qualquer pista)
    near_any = np.array([np.any(np.abs(d_t - x) <= tol_s) for x in g_t])
    out["onset_recall"] = round(float(near_any.mean()), 3)
    out["onset_precision"] = round(float(np.mean([np.any(np.abs(g_t - x) <= tol_s) for x in d_t])), 3)
    out["hit_recall"] = round(float(tp.mean()), 3)
    out["hit_precision"] = round(float(used.sum()) / max(1, d_t.size), 3)
    out["hit_f1"] = round(2 * out["hit_recall"] * out["hit_precision"] /
                          max(1e-9, out["hit_recall"] + out["hit_precision"]), 3)

    mi = matched_det[tp]
    gi = np.flatnonzero(tp)
    if mi.size:
        dbar_off = np.bincount((d_bar[mi] - g_bar[gi]).astype(int) + 64, minlength=129)
        sh = int(np.argmax(dbar_off)) - 64          # deslocamento modal de compasso
        d_off = sh
        dpos = np.abs(d_tick[mi] - g_tick[gi])
        same_bar = (d_bar[mi] - d_off) == (g_bar[gi] - g_off)
        wrap = np.minimum(dpos, 32 - dpos)
        pos_ok = same_bar & (wrap <= tick_tol)
        out["pos_accuracy"] = round(float(pos_ok.mean()), 3)
        out["tick_mae"] = round(float(np.mean(dpos[pos_ok])) if pos_ok.any() else -1, 3)
        out["time_bias_ms"] = round(float(np.mean(d_t[mi] - g_t[gi])) * 1000.0, 2)
        out["time_mae_ms"] = round(float(np.mean(np.abs(d_t[mi] - g_t[gi]))) * 1000.0, 2)
        vg, vd = g_vel[gi][pos_ok], d_vel[mi][pos_ok]
        if vg.size > 3:
            out["vel_pearson"] = round(float(np.corrcoef(vg, vd)[0, 1]), 3)
            out["vel_mae"] = round(float(np.mean(np.abs(vg - vd))), 1)
        out["pos_correct_of_gt"] = round(float(pos_ok.sum()) / max(1, g_t.size), 3)
    else:
        out.update({"pos_accuracy": 0.0, "pos_correct_of_gt": 0.0})

    # confusões de pista: GT sem par, mas com detecção no tempo
    conf = {}
    for i in np.flatnonzero(~tp):
        cand = np.flatnonzero(np.abs(d_t - g_t[i]) <= tol_s)
        if cand.size:
            j = int(cand[np.argmin(np.abs(d_t[cand] - g_t[i]))])
            k = f"{g_lane[i]}→{d_lane[j]}"
            conf[k] = conf.get(k, 0) + 1
    out["lane_confusion"] = dict(sorted(conf.items(), key=lambda kv: -kv[1])[:8])
    out["missed_only"] = int(np.sum(~tp) - sum(conf.values()))

    # por pista
    per = {}
    for ln in sorted(set(g_lane.tolist()) | set(d_lane.tolist())):
        gm = g_lane == ln
        dm = d_lane == ln
        m = np.flatnonzero(tp & (matched_det >= 0))
        ok = int(np.sum(dm[matched_det[m]] )) if m.size else 0
        per[ln] = {"gt": int(gm.sum()), "det": int(dm.sum()),
                   "recall": round(float(np.sum(tp[gm]) / max(1, int(gm.sum()))), 3)}
    out["per_lane"] = per
    out["bpm"] = score_dict.get("bpm")
    out["gt_bpm"] = gt["bpm"]
    out["bpm_err_pct"] = round(100.0 * abs(float(score_dict.get("bpm", 0)) - gt["bpm"]) / gt["bpm"], 3)
    out["meter"] = score_dict.get("meter")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(ROOT, "samples", "demo_drums.wav"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "samples", "demo_groundtruth.json"))
    ap.add_argument("--sens", type=float, default=0.95)
    ap.add_argument("--grid", default="auto")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    with open(args.file, "rb") as f:
        data = f.read()
    r = transcribe_bytes(data, filename=os.path.basename(args.file),
                         params={"sensitivity": args.sens, "grid_mode": args.grid})
    res = evaluate(r["score"].to_dict(), r["detections"], json.load(open(args.gt)))
    if args.json:
        print(json.dumps(res, indent=1, ensure_ascii=False))
    else:
        for k, v in res.items():
            print("  %-20s %s" % (k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
