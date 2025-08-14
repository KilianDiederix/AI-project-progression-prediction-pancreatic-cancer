#!/usr/bin/env python3
"""
Pancreatic cancer progression classifier (3D ResNet-34 + LSTM/Transformer patient sequence model)
Patient-level strict CV. Multi-cube, time-aware input.
Includes: TTA, z-score/minmax normalization, patient-level LSTM/Transformer over all cubes.
"""

import sys
sys.path.append(r"/...") #path to medicalnet
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchmetrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import random
import re, types
from collections import defaultdict, Counter
import pandas as pd
import os
import csv
import json
import datetime
import uuid
import matplotlib.pyplot as plt


repo_root = r"/..." # path to medicalnet again
sys.path.append(repo_root)
import models
sys.modules['medicalnet'] = types.ModuleType('medicalnet')
sys.modules['medicalnet.models'] = models
from models import generate_model #if added to medicalnet, all version i found online didn't include it, if it doesn't add:
# in MedicalNet/models/resnet.py add:
""""
def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if model_depth == 10:
        model = resnet10(**kwargs)
    elif model_depth == 18:
        model = resnet18(**kwargs)
    elif model_depth == 34:
        model = resnet34(**kwargs)
    elif model_depth == 50:
        model = resnet50(**kwargs)
    elif model_depth == 101:
        model = resnet101(**kwargs)
    elif model_depth == 152:
        model = resnet152(**kwargs)
    elif model_depth == 200:
        model = resnet200(**kwargs)
    return model
"""
# add __init__.py with: from .resnet import generate_model














# start

# Add progressive map for patients, removed for privacy reasons.
progressive_map = {
    "R001": x, "R002": x, "R003": x,# etc.
}

def extract_months_from_name(fname):
    # 1. Try -Xm or -XM or X_M 
    m = re.search(r'-?(\d+)[mM][-_]?(?:PRE)?', fname)
    if m:
        return int(m.group(1))
    # 2. If the file mentions "CTnonc", treat as baseline (0)
    if "CTnonc" in fname:
        return 0
    # 3. Fallback
    print(f"WARNING: Could not extract months from filename: {fname}")
    return 0



class PatientSequenceDataset(Dataset):
    def __init__(self, pid_to_cubes, progressive_map, augment=False):
        self.pids = list(pid_to_cubes.keys())
        self.cube_lists = []
        self.month_lists = []
        for pid in self.pids:
            files = pid_to_cubes[pid]
            # Pair each file with its month value
            cube_month_pairs = [(f, extract_months_from_name(f.name)) for f in files]
            # Find baselines: 0 or CTnonc; fallback to lowest month if not
            baselines = [x for x in cube_month_pairs if x[1] == 0]
            if baselines:
                # Baseline found, non-baselines are others
                non_baselines = [x for x in cube_month_pairs if x[1] != 0]
                # Sort non-baselines in descending month (most recent first), then baseline last
                cube_month_pairs_sorted = sorted(non_baselines, key=lambda x: -x[1]) + baselines
            else:
                # No baseline, so use the lowest month as fallback "baseline"
                if len(cube_month_pairs) == 0:
                    cube_month_pairs_sorted = []
                else:
                    min_month = min(x[1] for x in cube_month_pairs)
                    fallback = [x for x in cube_month_pairs if x[1] == min_month]
                    non_fallback = [x for x in cube_month_pairs if x[1] != min_month]
                    cube_month_pairs_sorted = sorted(non_fallback, key=lambda x: -x[1]) + fallback
            cubes_sorted = [x[0] for x in cube_month_pairs_sorted]
            months_sorted = [x[1] for x in cube_month_pairs_sorted]
            self.cube_lists.append(cubes_sorted)
            self.month_lists.append(months_sorted)


        self.labels = [progressive_map[pid] for pid in self.pids]
        self.augment = augment

        for pid, months, cubes in zip(self.pids, self.month_lists, self.cube_lists):
                print(f"Patient {pid}: months {months}, files {[f.name for f in cubes]}")

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        cubes = self.cube_lists[idx]
        months = self.month_lists[idx]
        label = self.labels[idx]
        pid = self.pids[idx]

        images = []
        for f in cubes:
            data = np.load(f)
            img = torch.as_tensor(data["image"][None], dtype=torch.float32)
            if self.augment:
                for axis in range(1, 4):
                    if torch.rand(1) < 0.5:
                        img = torch.flip(img, dims=[axis])
                if torch.rand(1) < 0.4:
                    img = img + torch.randn_like(img) * 0.03
            images.append(img)
        images = torch.stack(images)      # [num_cubes, 1, D, H, W]
        months = torch.tensor(months, dtype=torch.float32).unsqueeze(1) # / 12.0 --> removed div by 12.
        return images, months, torch.tensor(label, dtype=torch.float32), pid

def collate_patient_sequences(batch):
    # Pads sequences in batch to same length
    images, months, labels, pids = zip(*batch)
    max_len = max(img.shape[0] for img in images)
    pad_img = lambda img: torch.cat([img, img[-1:].repeat(max_len - img.shape[0], 1, 1, 1, 1)], dim=0) if img.shape[0] < max_len else img
    pad_month = lambda mon: torch.cat([mon, mon[-1:].repeat(max_len - mon.shape[0], 1)], dim=0) if mon.shape[0] < max_len else mon
    images = torch.stack([pad_img(img) for img in images])      # [B, max_seq, 1, D, H, W]
    months = torch.stack([pad_month(mon) for mon in months])    # [B, max_seq, 1]
    labels = torch.stack(labels)
    return images, months, labels, pids

# Optional transformer stub 
# class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1, n_heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    def forward(self, x):
        return self.transformer(x)

class ResNet34_SEQ_Classifier(pl.LightningModule):
    def __init__(self, weight_path: Path, seq_model="lstm", seq_hidden=128, seq_layers=1, n_heads=4, lr=1e-4, weight_decay=1e-4, dropout=0.5, frozen_blocks=3):
        super().__init__()
        self.save_hyperparameters(ignore=["weight_path"])
        self.seq_model_type = seq_model.lower()
        self.seq_hidden = seq_hidden
        self.seq_layers = seq_layers
        self.n_heads = n_heads
        self.backbone = generate_model(
            model_depth=34,
            sample_input_D=96,
            sample_input_H=96,
            sample_input_W=96,
            num_seg_classes=2,
            shortcut_type='B',
            no_cuda=False
        )
        state = torch.load(weight_path, map_location="cpu")
        if "state_dict" in state:
            state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state["state_dict"].items()}
        self.backbone.load_state_dict(state, strict=False)
        for name, p in self.backbone.named_parameters():
            if name.startswith("layer"):
                block_num = int(name.split(".")[0][-1])
                if block_num <= frozen_blocks:
                    p.requires_grad = False
                elif frozen_blocks == -1:
                    p.requires_grad = True
            elif name.startswith(("conv1", "bn1")):
                if frozen_blocks == -1:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        def backbone_feature_forward(this, x):
            x = this.conv1(x); x = this.bn1(x); x = this.relu(x); x = this.maxpool(x)
            x = this.layer1(x); x = this.layer2(x); x = this.layer3(x); x = this.layer4(x)
            x = nn.AdaptiveAvgPool3d(1)(x)
            x = x.view(x.size(0), -1)
            return x
        self.backbone.forward = backbone_feature_forward.__get__(self.backbone, type(self.backbone))

        # Sequence model selection
        seq_input_dim = 512+1
        if self.seq_model_type == "lstm":
            self.seq = nn.LSTM(input_size=seq_input_dim, hidden_size=seq_hidden, num_layers=seq_layers, batch_first=True, bidirectional=True)
            seq_out_dim = seq_hidden*2
        elif self.seq_model_type == "transformer":
            self.seq = TransformerEncoder(input_dim=seq_input_dim, hidden_dim=seq_hidden, n_layers=seq_layers, n_heads=n_heads, dropout=dropout) # uncomment transformer section, but this should probably just be removed, transformers didn't work.
            seq_out_dim = seq_input_dim
        else:
            raise ValueError("seq_model must be 'lstm' or 'transformer'")
        self.head = nn.Sequential(
            nn.Linear(seq_out_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")

        # --- Added for logging ---
        self.train_auc_list = []
        self.val_auc_list = []
        self.train_loss_list = []
        self.val_loss_list = []

    def forward(self, images, months):
        B, T, C, D, H, W = images.shape
        print("Images:", images.shape, images.mean().item(), images.std().item())
        print("Months:", months)

        feats = []
        for t in range(T):
            x = images[:, t] # [B, 1, D, H, W]
            f = self.backbone(x)     # [B, 512]
            # print("Backbone features:", f[0, :10].detach().cpu().numpy())  # first 10 elements
            m = months[:, t]         # [B, 1]
            f = torch.cat([f, m], dim=1) # [B, 513]
            feats.append(f)
        feats = torch.stack(feats, dim=1)    # [B, T, 513]
        if self.seq_model_type == "lstm":
            seq_out, _ = self.seq(feats)  # [B, T, hidden*2]
            patient_vec = seq_out[:, -1, :]  # last timepoint
        else:
            seq_out = self.seq(feats)  # [B, T, 513]
            patient_vec = seq_out[:, -1, :]  # last timepoint
        out = self.head(patient_vec)
        return out.squeeze(-1)

    def training_step(self, batch, _):
        images, months, y, _ = batch
        images = images.to(self.device)
        months = months.to(self.device)
        y = y.to(self.device)
        logits = self(images, months)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.train_auc.update(probs, y.long())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        auc = self.train_auc.compute().item()
        loss = self.trainer.callback_metrics["train_loss"].item()
        self.train_auc_list.append(auc)
        self.train_loss_list.append(loss)
        self.log("train_auc", auc, prog_bar=True)
        self.train_auc.reset()

    def validation_step(self, batch, _):
        images, months, y, pids = batch
        images = images.to(self.device)
        months = months.to(self.device)
        y = y.to(self.device)
        logits = self(images, months)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        self.val_auc.update(probs, y.long())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_auc", self.val_auc, prog_bar=True)
        # --- Added for logging ---
        self.val_loss = loss.detach().cpu().item()
        return {"probs": probs.cpu().numpy(), "labels": y.cpu().numpy(), "pids": pids, "loss": loss}

    def on_validation_epoch_end(self):
        try:
            auc = self.val_auc.compute().item()
        except Exception:
            auc = float("nan")
        # Log for plotting
        self.val_auc_list.append(auc)
        # Need to grab last val_loss (since val_loss is averaged in the trainer)
        if hasattr(self, "val_loss"):
            self.val_loss_list.append(self.val_loss)
        else:
            self.val_loss_list.append(float("nan"))
        self.log("val_auc_epoch", auc, prog_bar=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                               lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=8)
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val_auc_epoch"}

def run_cv(cube_dir: Path, weight_path: Path, batch: int, epochs: int, folds: int = 5,
           lr=1e-4, weight_decay=1e-4, frozen_blocks=3, dropout=0.5, output_dir=".",
           seq_model="lstm", seq_hidden=128, seq_layers=1, n_heads=4, tta=6):

    pid_to_cubes = defaultdict(list)
    for f in cube_dir.glob("*.npz"):
        pid = f.name.split("_")[0]
        if pid in progressive_map:
            pid_to_cubes[pid].append(f)
    pids = sorted(pid_to_cubes.keys())
    y = [progressive_map[pid] for pid in pids]
    print("N PIDs:", len(pids), "N labels:", len(y))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    summary_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(pids, y), 1):
        print(f"\n=== Fold {fold}/{folds} ===")
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        exp_name = f"{seq_model}_cv_lr{lr}_wd{weight_decay}_fb{frozen_blocks}_dp{dropout}_{now}_{unique_id}"
        fold_dir = Path(output_dir) / exp_name / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        tr_pids = [pids[i] for i in tr_idx]
        va_pids = [pids[i] for i in va_idx]
        tr_pid_cubes = {pid: pid_to_cubes[pid] for pid in tr_pids}
        va_pid_cubes = {pid: pid_to_cubes[pid] for pid in va_pids}

        print(f"Train PIDs: {len(tr_pids)}, Val PIDs: {len(va_pids)}")
        tr_ds = PatientSequenceDataset(tr_pid_cubes, progressive_map, augment=False)
        va_ds = PatientSequenceDataset(va_pid_cubes, progressive_map, augment=False)
        print(f"Train dataset length: {len(tr_ds)}, Val dataset length: {len(va_ds)}")

        tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_patient_sequences)
        va_dl = DataLoader(va_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_patient_sequences)

        model = ResNet34_SEQ_Classifier(
            weight_path=weight_path,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            frozen_blocks=frozen_blocks,
            seq_model=seq_model,
            seq_hidden=seq_hidden,
            seq_layers=seq_layers,
            n_heads=n_heads
        )
        print("\n=== Model parameter requires_grad status ===")
        for name, param in model.named_parameters():
            print(f"{name:60s}  requires_grad={param.requires_grad}")
        print("\n=== Frozen parameters ===")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(name)


        early_stop = pl.callbacks.EarlyStopping(monitor="val_auc_epoch", mode="max", patience=15, verbose=True)
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            default_root_dir=str(fold_dir),
            log_every_n_steps=10,
            enable_checkpointing=False,
            callbacks=[early_stop],
            num_sanity_val_steps=0
        )
        trainer.fit(model, tr_dl, va_dl)

        # -------------------- Evaluation
        model.eval()
        patient_preds, patient_trues, patient_ids = [], [], []
        with torch.no_grad():
            for (images, months, y, pid) in va_dl:
                images = images.to(model.head[0].weight.device)
                months = months.to(model.head[0].weight.device)
                logits = model(images, months)
                prob = torch.sigmoid(logits).cpu().numpy()
                patient_preds.append(prob)
                patient_trues.append(y.cpu().numpy())
                patient_ids.append(pid[0])

        scan_auc = float('nan') # only patient-level, since one prediction per patient
        if len(set(np.array(patient_trues).flatten())) > 1:
            pat_auc = roc_auc_score(patient_trues, patient_preds)
        else:
            pat_auc = float("nan")

        print(f"[Fold {fold}] Patient-level AUC: {pat_auc:.3f}")

        pd.DataFrame({
            "pid": patient_ids,
            "prob": np.array(patient_preds).flatten(),
            "label": np.array(patient_trues).flatten(),
            "fold": fold
        }).to_csv(fold_dir / "patient_preds.csv", index=False)

        # --- Added: Save logs and plot metrics ---
        logs = {
            "train_aucs": model.train_auc_list,
            "val_aucs": model.val_auc_list,
            "train_losses": model.train_loss_list,
            "val_losses": model.val_loss_list
        }
        with open(fold_dir / "logs.json", "w") as f:
            json.dump(logs, f, indent=2)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(logs["train_aucs"], label="Train AUC")
        ax[0].plot(logs["val_aucs"], label="Val AUC")
        ax[0].set_title(f"AUCROC (Fold {fold})")
        ax[0].legend(); ax[0].set_xlabel("Epoch")
        ax[1].plot(logs["train_losses"], label="Train Loss")
        ax[1].plot(logs["val_losses"], label="Val Loss")
        ax[1].set_title(f"Loss (Fold {fold})")
        ax[1].legend(); ax[1].set_xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(str(fold_dir / "metrics.png"))
        plt.close()
        # --- End logs/plot add ---

        summary_rows.append({
            "fold": fold,
            "scan_auc": scan_auc,
            "patient_auc": pat_auc,
            "n_train_patients": len(tr_pids),
            "n_val_patients": len(va_pids),
        })
        all_preds = []
        all_labels = []
        for fold in range(1, folds+1):
            fold_dir = Path(output_dir) / exp_name / f"fold{fold}"
            csv_path = fold_dir / "patient_preds.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_preds.extend(df['prob'].values)
                all_labels.extend(df['label'].values)

        if all_preds:
            plt.figure(figsize=(6, 6))
            plt.scatter(all_preds, all_labels, alpha=0.7)
            plt.xlabel("Predicted Probability")
            plt.ylabel("True Label")
            plt.title("Patient Predictions: Prob vs. Label")
            plt.yticks([0, 1])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(str(Path(output_dir) / exp_name / "preds_vs_labels.png"))
            plt.close()
            print(f"Saved scatter plot to {Path(output_dir) / exp_name / 'preds_vs_labels.png'}")
        else:
            print("No predictions to plot.")
    # --- Write summary CSV (all folds)
    summary_path = Path(output_dir) / exp_name / "cv_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline='') as csvfile:
        fieldnames = summary_rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\n=== CV summary ===")
    for row in summary_rows:
        print(f"Fold {row['fold']}: Patient AUC={row['patient_auc']:.3f}")
    print(f"Mean Patient AUC: {np.nanmean([row['patient_auc'] for row in summary_rows]):.3f} ± {np.nanstd([row['patient_auc'] for row in summary_rows]):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube_dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--frozen_blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--output_dir", type=str, default=".")
    ap.add_argument("--seq_model", type=str, default="lstm", choices=["lstm", "transformer"], help="Type of sequence model")
    ap.add_argument("--seq_hidden", type=int, default=128, help="Hidden size for sequence model")
    ap.add_argument("--seq_layers", type=int, default=1, help="Number of layers for sequence model")
    ap.add_argument("--n_heads", type=int, default=4, help="Number of heads for transformer (if used)")
    ap.add_argument("--tta", type=int, default=6, help="Number of Test-Time Augmentations (TTA) per scan")
    args = ap.parse_args()
    run_cv(
        cube_dir=args.cube_dir,
        weight_path=args.weights,
        batch=args.batch,
        epochs=args.epochs,
        folds=args.folds,
        lr=args.lr,
        weight_decay=args.weight_decay,
        frozen_blocks=args.frozen_blocks,
        dropout=args.dropout,
        output_dir=args.output_dir,
        seq_model=args.seq_model,
        seq_hidden=args.seq_hidden,
        seq_layers=args.seq_layers,
        n_heads=args.n_heads,
        tta=args.tta,
    )
