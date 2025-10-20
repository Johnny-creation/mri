#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search ONLY for T and beta (Path-A spiking readout).
- 固定其他训练超参（见下方常量），只搜索 T 与 beta。
- 依赖：你已按路径A改好代码；pip install snntorch
"""

import argparse, itertools, os, csv, time, traceback
import torch

from utils import set_seed
from dataset import get_multitask_loaders
from model import MultiTaskModel
from train import train_multitask
from evaluate import evaluate_multitask

# ===== 固定不变的训练设置（需要时你可改） =====
FIXED = {
    "batch_size": 32,
    "epochs": 70,
    "lr": 8e-4,
    "dropout_rate": 0.4,
    "weight_decay": 0.0,
    "time_loss_weight": 1.0,
    "seed": 42,
    "log_dir": "output/sweep_Tbeta"
}
# ==============================================

def parse_list(s, cast):
    return [cast(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser("Grid search for T & beta (SNN readout)")
    ap.add_argument("--base_path", type=str, default="../data/lesion-selected")
    ap.add_argument("--grid_T", type=str, default="6,10,14", help="comma list, e.g. 6,10,14")
    ap.add_argument("--grid_beta", type=str, default="0.85,0.9,0.95", help="comma list, e.g. 0.85,0.9,0.95")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", type=str, default="sweep_Tbeta_results.csv")
    ap.add_argument("--max_runs", type=int, default=0, help="0=all; >0 limit for quick try")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(FIXED["log_dir"], exist_ok=True)

    grid_T = parse_list(args.grid_T, int)
    grid_beta = parse_list(args.grid_beta, float)
    combos = list(itertools.product(grid_T, grid_beta))
    if args.max_runs and args.max_runs > 0:
        combos = combos[:args.max_runs]

    new_file = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "timestamp","T","beta","seed","epochs","lr","dropout",
                "val_lesion_acc","val_lesion_auc","val_lesion_f1",
                "val_time_acc","val_time_auc","val_time_f1",
                "log_file","final_model_path","best_model_path"
            ])

    print(f"[Sweep] runs: {len(combos)} (T x beta)")
    best = None  # (score_tuple, row_dict)

    for i, (T, beta) in enumerate(combos, 1):
        print(f"\n=== Run {i}/{len(combos)}: T={T}, beta={beta} ===")
        try:
            set_seed(FIXED["seed"])
            device = torch.device(args.device)

            train_loader, val_loader, _ = get_multitask_loaders(args.base_path, FIXED["batch_size"])

            model = MultiTaskModel(
                dropout_rate=FIXED["dropout_rate"],
                use_snn_head=True,
                T=T,
                beta=beta
            ).to(device)

            log_file, final_model_path, best_model_path, _maybe_val = train_multitask(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=FIXED["epochs"],
                lr=FIXED["lr"],
                dropout_rate=FIXED["dropout_rate"],
                weight_decay=FIXED["weight_decay"],
                time_loss_weight=FIXED["time_loss_weight"],
                log_dir=FIXED["log_dir"]
            )

            val_metrics = evaluate_multitask(model, val_loader, device=device, log_file=log_file)

            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "T": T, "beta": beta, "seed": FIXED["seed"],
                "epochs": FIXED["epochs"], "lr": FIXED["lr"], "dropout": FIXED["dropout_rate"],
                "val_lesion_acc": val_metrics.get("lesion_acc"),
                "val_lesion_auc": val_metrics.get("lesion_auc"),
                "val_lesion_f1":  val_metrics.get("lesion_f1"),
                "val_time_acc":   val_metrics.get("time_acc"),
                "val_time_auc":   val_metrics.get("time_auc"),
                "val_time_f1":    val_metrics.get("time_f1"),
                "log_file": log_file,
                "final_model_path": final_model_path,
                "best_model_path": best_model_path
            }

            with open(args.out_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    row["timestamp"], row["T"], row["beta"], row["seed"], row["epochs"], row["lr"], row["dropout"],
                    row["val_lesion_acc"], row["val_lesion_auc"], row["val_lesion_f1"],
                    row["val_time_acc"], row["val_time_auc"], row["val_time_f1"],
                    row["log_file"], row["final_model_path"], row["best_model_path"]
                ])

            # 以 time_auc 为主、lesion_auc 为次的排序
            score = (
                row["val_time_auc"] if row["val_time_auc"] is not None else -1.0,
                row["val_lesion_auc"] if row["val_lesion_auc"] is not None else -1.0
            )
            if (best is None) or (score > best[0]):
                best = (score, row)
                print(f"[Best so far] time_auc={row['val_time_auc']:.4f} | lesion_auc={row['val_lesion_auc']:.4f}")

        except Exception as e:
            print(f"[Run {i}] ERROR: {e}")
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best is None:
        print("\nNo successful runs.")
    else:
        br = best[1]
        print("\n=== Best config (by val time AUC, then lesion AUC) ===")
        print(f"T={br['T']}, beta={br['beta']}")
        print(f"time_auc={br['val_time_auc']}, lesion_auc={br['val_lesion_auc']}")
        print(f"log: {br['log_file']}")
        print(f"final: {br['final_model_path']}")
        print(f"best(ckpt): {br['best_model_path']}")

if __name__ == "__main__":
    main()
