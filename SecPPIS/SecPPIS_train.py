# -*- coding: utf-8 -*-

import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
)

from tqdm import tqdm

AA_DIR = "aa_sequences"
SS_DIR = "secondary_structures"
LABEL_DIR = "labels"

BASE_OUTPUT_DIR = "output_secppis_feature_mlp_5runs"

SEEDS = [42, 2024, 2025, 3407, 10086]

NUM_EPOCHS = 35

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

WINDOW_SIZE = 15
HALF_WINDOW = WINDOW_SIZE // 2

BATCH_SIZE = 256

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SAMPLER_POWER = 0.5

DROPOUT = 0.4

AA_DIM = 21   # 20 standard AA + unknown X
SS_DIM = 8    # DSSP 0-7

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_protein_id(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def build_protein_metadata(aa_dir, ss_dir, label_dir):
    aa_files = set([f for f in os.listdir(aa_dir) if f.endswith(".npy")])
    ss_files = set([f for f in os.listdir(ss_dir) if f.endswith(".npy")])
    label_files = set([f for f in os.listdir(label_dir) if f.endswith(".npy")])

    common_files = sorted(list(aa_files & ss_files & label_files))

    if len(common_files) == 0:
        raise ValueError("No matched aa/ss/label files found.")

    all_files = aa_files | ss_files | label_files
    skipped_missing = len(all_files - set(common_files))

    metadata = []
    adjusted_mismatch = 0

    total_residues = 0
    total_0 = 0
    total_1 = 0

    for file_name in common_files:
        protein_id = extract_protein_id(file_name)

        aa_path = os.path.join(aa_dir, file_name)
        ss_path = os.path.join(ss_dir, file_name)
        label_path = os.path.join(label_dir, file_name)

        aa = np.load(aa_path).astype(int).reshape(-1)
        ss = np.load(ss_path).astype(int).reshape(-1)
        labels = np.load(label_path).astype(int).reshape(-1)

        if not (len(aa) == len(ss) == len(labels)):
            print(
                f"Length mismatch: {protein_id}, "
                f"aa={len(aa)}, ss={len(ss)}, labels={len(labels)}. "
                f"Truncating to shortest length."
            )
            n = min(len(aa), len(ss), len(labels))
            aa = aa[:n]
            ss = ss[:n]
            labels = labels[:n]
            adjusted_mismatch += 1
        else:
            n = len(labels)

        if n == 0:
            continue

        if np.any((aa < 0) | (aa > 20)):
            raise ValueError(f"AA values must be 0-20 in {aa_path}")

        if np.any((ss < 0) | (ss > 7)):
            raise ValueError(f"SS values must be 0-7 in {ss_path}")

        if np.any((labels != 0) & (labels != 1)):
            raise ValueError(f"Labels must be 0 or 1 in {label_path}")

        label_0 = int(np.sum(labels == 0))
        label_1 = int(np.sum(labels == 1))

        metadata.append({
            "protein_id": protein_id,
            "aa_path": aa_path,
            "ss_path": ss_path,
            "label_path": label_path,
            "length": n,
            "label_0": label_0,
            "label_1": label_1,
        })

        total_residues += n
        total_0 += label_0
        total_1 += label_1

    print("=" * 90)
    print("Input file summary")
    print("=" * 90)
    print(f"Matched proteins: {len(metadata)}")
    print(f"Files skipped due to missing counterpart: {skipped_missing}")
    print(f"Adjusted length mismatch files: {adjusted_mismatch}")
    print(f"Total residues: {total_residues}")
    print(f"Label 0: {total_0}")
    print(f"Label 1: {total_1}")
    print(f"Positive ratio: {total_1 / (total_0 + total_1):.4f}")

    return metadata


def split_by_protein(metadata, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    protein_ids = sorted([m["protein_id"] for m in metadata])

    random.seed(seed)
    random.shuffle(protein_ids)

    n_total = len(protein_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_proteins = set(protein_ids[:n_train])
    val_proteins = set(protein_ids[n_train:n_train + n_val])
    test_proteins = set(protein_ids[n_train + n_val:])

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, m in enumerate(metadata):
        if m["protein_id"] in train_proteins:
            train_indices.append(idx)
        elif m["protein_id"] in val_proteins:
            val_indices.append(idx)
        elif m["protein_id"] in test_proteins:
            test_indices.append(idx)
        else:
            raise RuntimeError("Protein ID not assigned.")

    return train_indices, val_indices, test_indices, train_proteins, val_proteins, test_proteins


def save_split_metadata(metadata, train_indices, val_indices, test_indices, output_dir):
    split_map = {}

    for idx in train_indices:
        split_map[idx] = "train"
    for idx in val_indices:
        split_map[idx] = "validation"
    for idx in test_indices:
        split_map[idx] = "test"

    split_file = os.path.join(output_dir, "protein_level_split_metadata.csv")

    with open(split_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "protein_id",
            "length",
            "label_0",
            "label_1",
            "split",
            "aa_path",
            "ss_path",
            "label_path"
        ])

        for idx, m in enumerate(metadata):
            writer.writerow([
                m["protein_id"],
                m["length"],
                m["label_0"],
                m["label_1"],
                split_map[idx],
                m["aa_path"],
                m["ss_path"],
                m["label_path"]
            ])

    print(f"Saved split metadata to: {split_file}")


def summarize_split(metadata, indices, split_name):
    total_len = sum(metadata[i]["length"] for i in indices)
    total_0 = sum(metadata[i]["label_0"] for i in indices)
    total_1 = sum(metadata[i]["label_1"] for i in indices)

    print(f"\n[{split_name}]")
    print(f"Proteins: {len(indices)}")
    print(f"Residues: {total_len}")
    print(f"Label 0: {total_0}")
    print(f"Label 1: {total_1}")
    print(f"Positive ratio: {total_1 / (total_0 + total_1):.4f}")


def build_residue_index(metadata, protein_indices):
    residue_index = []

    for protein_idx in protein_indices:
        length = metadata[protein_idx]["length"]
        for residue_idx in range(length):
            residue_index.append((protein_idx, residue_idx))

    return residue_index


HYDROPHOBIC = set([0, 4, 7, 9, 10, 12, 13, 14, 17, 18, 19])  # A,C,G,I,L,M,F,P,W,Y,V
POSITIVE = set([1, 8, 11])  # R,H,K
NEGATIVE = set([3, 6])      # D,E
POLAR = set([2, 5, 15, 16]) # N,Q,S,T
AROMATIC = set([13, 17, 18]) # F,W,Y
SPECIAL = set([4, 7, 14])   # C,G,P


def one_hot(index, dim):
    arr = np.zeros(dim, dtype=np.float32)
    if 0 <= index < dim:
        arr[index] = 1.0
    return arr


def normalized_counts(values, dim):
    counts = np.zeros(dim, dtype=np.float32)

    valid = [v for v in values if 0 <= v < dim]

    if len(valid) == 0:
        return counts

    for v in valid:
        counts[v] += 1.0

    counts = counts / float(len(valid))
    return counts


def aa_group_features(aa_window):
    valid = [int(a) for a in aa_window if 0 <= int(a) <= 20]

    if len(valid) == 0:
        return np.zeros(6, dtype=np.float32)

    n = float(len(valid))

    hydrophobic = sum([1 for a in valid if a in HYDROPHOBIC]) / n
    positive = sum([1 for a in valid if a in POSITIVE]) / n
    negative = sum([1 for a in valid if a in NEGATIVE]) / n
    polar = sum([1 for a in valid if a in POLAR]) / n
    aromatic = sum([1 for a in valid if a in AROMATIC]) / n
    special = sum([1 for a in valid if a in SPECIAL]) / n

    return np.array([
        hydrophobic,
        positive,
        negative,
        polar,
        aromatic,
        special
    ], dtype=np.float32)


def ss_run_length(ss, residue_idx):
    center_ss = int(ss[residue_idx])
    length = len(ss)

    left = residue_idx
    while left - 1 >= 0 and int(ss[left - 1]) == center_ss:
        left -= 1

    right = residue_idx
    while right + 1 < length and int(ss[right + 1]) == center_ss:
        right += 1

    run_len = right - left + 1
    return run_len


def make_residue_feature(aa, ss, residue_idx, window_size=15):
    length = len(aa)
    half = window_size // 2

    center_aa = int(aa[residue_idx])
    center_ss = int(ss[residue_idx])

    aa_window = []
    ss_window = []

    for pos in range(residue_idx - half, residue_idx + half + 1):
        if 0 <= pos < length:
            aa_window.append(int(aa[pos]))
            ss_window.append(int(ss[pos]))

    center_aa_oh = one_hot(center_aa, AA_DIM)
    center_ss_oh = one_hot(center_ss, SS_DIM)

    aa_comp = normalized_counts(aa_window, AA_DIM)
    ss_comp = normalized_counts(ss_window, SS_DIM)

    aa_groups = aa_group_features(aa_window)

    if length > 1:
        rel_pos = residue_idx / float(length - 1)
    else:
        rel_pos = 0.0

    dist_n = rel_pos
    dist_c = 1.0 - rel_pos
    protein_len_norm = min(length / 1000.0, 1.0)

    position_features = np.array([
        rel_pos,
        dist_n,
        dist_c,
        protein_len_norm
    ], dtype=np.float32)

    run_len = ss_run_length(ss, residue_idx)
    run_len_norm = min(run_len / 50.0, 1.0)

    run_feature = np.array([run_len_norm], dtype=np.float32)

    feature = np.concatenate([
        center_aa_oh,      # 21
        center_ss_oh,      # 8
        aa_comp,           # 21
        ss_comp,           # 8
        aa_groups,         # 6
        position_features, # 4
        run_feature        # 1
    ])

    return feature.astype(np.float32)


class FeatureMLPDataset(Dataset):
    def __init__(self, metadata, residue_index, window_size=15):
        self.metadata = metadata
        self.residue_index = residue_index
        self.window_size = window_size
        self.cache = {}

    def __len__(self):
        return len(self.residue_index)

    def _load_protein(self, protein_idx):
        if protein_idx in self.cache:
            return self.cache[protein_idx]

        item = self.metadata[protein_idx]

        aa = np.load(item["aa_path"]).astype(int).reshape(-1)
        ss = np.load(item["ss_path"]).astype(int).reshape(-1)
        labels = np.load(item["label_path"]).astype(int).reshape(-1)

        if not (len(aa) == len(ss) == len(labels)):
            n = min(len(aa), len(ss), len(labels))
            aa = aa[:n]
            ss = ss[:n]
            labels = labels[:n]

        self.cache[protein_idx] = (aa, ss, labels)
        return aa, ss, labels

    def __getitem__(self, idx):
        protein_idx, residue_idx = self.residue_index[idx]

        aa, ss, labels = self._load_protein(protein_idx)

        feature = make_residue_feature(
            aa,
            ss,
            residue_idx,
            window_size=self.window_size
        )

        label = int(labels[residue_idx])

        return (
            torch.tensor(feature, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

def create_weighted_sampler(metadata, residue_index, power=0.5, seed=42):
    labels = []

    protein_label_cache = {}

    for protein_idx, residue_idx in residue_index:
        if protein_idx not in protein_label_cache:
            item = metadata[protein_idx]

            lab = np.load(item["label_path"]).astype(int).reshape(-1)
            aa = np.load(item["aa_path"]).astype(int).reshape(-1)
            ss = np.load(item["ss_path"]).astype(int).reshape(-1)

            n = min(len(aa), len(ss), len(lab))
            lab = lab[:n]

            protein_label_cache[protein_idx] = lab

        labels.append(int(protein_label_cache[protein_idx][residue_idx]))

    labels = np.array(labels, dtype=int)

    class_counts = np.bincount(labels, minlength=2)

    print("\nTraining class counts for WeightedRandomSampler:")
    print(f"Label 0: {class_counts[0]}")
    print(f"Label 1: {class_counts[1]}")

    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError("Both classes must exist in training set.")

    class_sampling_weights = (1.0 / class_counts) ** power
    sample_weights = np.array([class_sampling_weights[label] for label in labels])
    sample_weights = torch.DoubleTensor(sample_weights)

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator
    )

    return sampler


def check_sampler_distribution(train_sampler, metadata, train_residue_index, max_samples=5000):
    sampled_indices = list(iter(train_sampler))[:max_samples]

    label_cache = {}
    sampled_labels = []

    for sampled_idx in sampled_indices:
        protein_idx, residue_idx = train_residue_index[sampled_idx]

        if protein_idx not in label_cache:
            lab = np.load(metadata[protein_idx]["label_path"]).astype(int).reshape(-1)
            aa = np.load(metadata[protein_idx]["aa_path"]).astype(int).reshape(-1)
            ss = np.load(metadata[protein_idx]["ss_path"]).astype(int).reshape(-1)

            n = min(len(aa), len(ss), len(lab))
            lab = lab[:n]

            label_cache[protein_idx] = lab

        sampled_labels.append(int(label_cache[protein_idx][residue_idx]))

    count_0 = sampled_labels.count(0)
    count_1 = sampled_labels.count(1)
    total = count_0 + count_1

    print("\nApproximate sampled training distribution:")
    print(f"Sampled Label 0: {count_0}")
    print(f"Sampled Label 1: {count_1}")
    print(f"Sampled positive ratio: {count_1 / total:.4f}")


class SecFeatureMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.4):
        super(SecFeatureMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, dataloader, device, criterion, threshold=0.5, return_probs=False):
    model.eval()

    y_true = []
    y_prob = []

    total_loss = 0.0

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]

            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(y_true, y_pred)
    precision_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc = np.nan
        auprc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "balanced_accuracy": balanced_acc,
        "auroc": auroc,
        "auprc": auprc,
        "mcc": mcc,
        "confusion_matrix": cm,
        "threshold": threshold,
    }

    if return_probs:
        metrics["y_true"] = y_true
        metrics["y_prob"] = y_prob

    return metrics


def find_best_threshold(y_true, y_prob, metric="mcc"):
    thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = 0.5
    best_score = -999

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == "f1":
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric must be 'mcc', 'f1', or 'balanced_accuracy'")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0

    for features, labels in tqdm(dataloader, desc="Training", leave=False):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def run_one_seed(metadata, seed):
    print("\n" + "=" * 90)
    print(f"Running seed: {seed}")
    print("=" * 90)

    set_seed(seed)

    output_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    train_indices, val_indices, test_indices, train_proteins, val_proteins, test_proteins = split_by_protein(
        metadata,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=seed
    )

    save_split_metadata(metadata, train_indices, val_indices, test_indices, output_dir)

    summarize_split(metadata, train_indices, "Train")
    summarize_split(metadata, val_indices, "Validation")
    summarize_split(metadata, test_indices, "Independent Test")

    assert train_proteins.isdisjoint(val_proteins)
    assert train_proteins.isdisjoint(test_proteins)
    assert val_proteins.isdisjoint(test_proteins)

    print("\nProtein-level split check passed: no protein overlap among train, validation, and test sets.")

    train_residue_index = build_residue_index(metadata, train_indices)
    val_residue_index = build_residue_index(metadata, val_indices)
    test_residue_index = build_residue_index(metadata, test_indices)

    train_dataset = FeatureMLPDataset(
        metadata,
        residue_index=train_residue_index,
        window_size=WINDOW_SIZE
    )

    val_dataset = FeatureMLPDataset(
        metadata,
        residue_index=val_residue_index,
        window_size=WINDOW_SIZE
    )

    test_dataset = FeatureMLPDataset(
        metadata,
        residue_index=test_residue_index,
        window_size=WINDOW_SIZE
    )

    sample_feature, _ = train_dataset[0]
    input_dim = sample_feature.shape[0]
    print(f"\nFeature dimension: {input_dim}")

    train_sampler = create_weighted_sampler(
        metadata,
        train_residue_index,
        power=SAMPLER_POWER,
        seed=seed
    )

    check_sampler_distribution(train_sampler, metadata, train_residue_index)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    model = SecFeatureMLP(
        input_dim=input_dim,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    log_file = os.path.join(output_dir, "training_log.csv")

    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "learning_rate",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_precision_pos",
            "val_recall_pos",
            "val_f1_pos",
            "val_balanced_accuracy",
            "val_auroc",
            "val_auprc",
            "val_mcc",
            "val_threshold"
        ])

    best_val_auprc = -1
    best_model_path = os.path.join(output_dir, "best_model_by_val_auprc.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nSeed {seed} | Epoch {epoch}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            threshold=0.5,
            return_probs=False
        )

        scheduler.step(val_metrics["auprc"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Current learning rate: {current_lr:.8f}")
        print(
            f"Validation | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Precision+: {val_metrics['precision_pos']:.4f} | "
            f"Recall+: {val_metrics['recall_pos']:.4f} | "
            f"F1+: {val_metrics['f1_pos']:.4f} | "
            f"AUPRC: {val_metrics['auprc']:.4f} | "
            f"AUROC: {val_metrics['auroc']:.4f} | "
            f"MCC: {val_metrics['mcc']:.4f}"
        )

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{current_lr:.8f}",
                f"{train_loss:.6f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['accuracy']:.6f}",
                f"{val_metrics['precision_pos']:.6f}",
                f"{val_metrics['recall_pos']:.6f}",
                f"{val_metrics['f1_pos']:.6f}",
                f"{val_metrics['balanced_accuracy']:.6f}",
                f"{val_metrics['auroc']:.6f}",
                f"{val_metrics['auprc']:.6f}",
                f"{val_metrics['mcc']:.6f}",
                f"{val_metrics['threshold']:.2f}"
            ])

        if not np.isnan(val_metrics["auprc"]) and val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved based on validation AUPRC: {best_val_auprc:.4f}")

    print("\nLoading best model for final independent test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_metrics_for_threshold = evaluate(
        model,
        val_loader,
        device,
        criterion,
        threshold=0.5,
        return_probs=True
    )

    best_threshold, best_threshold_score = find_best_threshold(
        val_metrics_for_threshold["y_true"],
        val_metrics_for_threshold["y_prob"],
        metric="mcc"
    )

    print(f"\nBest threshold selected on validation set: {best_threshold:.2f}")
    print(f"Best validation MCC at selected threshold: {best_threshold_score:.4f}")

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        criterion,
        threshold=best_threshold,
        return_probs=False
    )

    print("\nFinal Independent Test Results")
    print("===================================")
    print(f"Seed: {seed}")
    print(f"Selected threshold from validation set: {best_threshold:.2f}")
    print(f"Validation MCC at selected threshold: {best_threshold_score:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Positive Precision: {test_metrics['precision_pos']:.4f}")
    print(f"Positive Recall: {test_metrics['recall_pos']:.4f}")
    print(f"Positive F1-score: {test_metrics['f1_pos']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"AUROC: {test_metrics['auroc']:.4f}")
    print(f"AUPRC: {test_metrics['auprc']:.4f}")
    print(f"MCC: {test_metrics['mcc']:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(test_metrics["confusion_matrix"])

    result = {
        "seed": seed,
        "best_val_auprc": best_val_auprc,
        "selected_threshold": best_threshold,
        "val_mcc_at_threshold": best_threshold_score,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_pos": test_metrics["precision_pos"],
        "test_recall_pos": test_metrics["recall_pos"],
        "test_f1_pos": test_metrics["f1_pos"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_auroc": test_metrics["auroc"],
        "test_auprc": test_metrics["auprc"],
        "test_mcc": test_metrics["mcc"],
        "tn": int(test_metrics["confusion_matrix"][0, 0]),
        "fp": int(test_metrics["confusion_matrix"][0, 1]),
        "fn": int(test_metrics["confusion_matrix"][1, 0]),
        "tp": int(test_metrics["confusion_matrix"][1, 1]),
    }

    test_result_file = os.path.join(output_dir, "final_independent_test_results.txt")

    with open(test_result_file, "w", encoding="utf-8") as f:
        f.write("Final Independent Test Results\n")
        f.write("===================================\n")
        f.write(f"Seed: {seed}\n")
        f.write("Model: SecPPIS engineered local AA+SS feature MLP\n")
        f.write(f"Window size: {WINDOW_SIZE}\n")
        f.write("Labels: original residue-level labels, no label expansion\n")
        f.write("Features: center AA/SS one-hot, local AA/SS composition, AA group composition, position, SS run length\n")
        f.write("Split: protein-level train/validation/independent test\n")
        f.write(f"Sampler power: {SAMPLER_POWER}\n")
        f.write(f"Best validation AUPRC: {best_val_auprc:.6f}\n")
        f.write(f"Selected threshold from validation set: {best_threshold:.2f}\n")
        f.write(f"Validation MCC at selected threshold: {best_threshold_score:.6f}\n")
        f.write(f"Test Loss: {test_metrics['loss']:.6f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"Positive Precision: {test_metrics['precision_pos']:.6f}\n")
        f.write(f"Positive Recall: {test_metrics['recall_pos']:.6f}\n")
        f.write(f"Positive F1-score: {test_metrics['f1_pos']:.6f}\n")
        f.write(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.6f}\n")
        f.write(f"AUROC: {test_metrics['auroc']:.6f}\n")
        f.write(f"AUPRC: {test_metrics['auprc']:.6f}\n")
        f.write(f"MCC: {test_metrics['mcc']:.6f}\n")
        f.write("Confusion Matrix [[TN, FP], [FN, TP]]:\n")
        f.write(str(test_metrics["confusion_matrix"]))
        f.write("\n")

    return result


def save_run_summary(results, output_dir):
    all_runs_file = os.path.join(output_dir, "all_runs_results.csv")

    fieldnames = [
        "seed",
        "best_val_auprc",
        "selected_threshold",
        "val_mcc_at_threshold",
        "test_loss",
        "test_accuracy",
        "test_precision_pos",
        "test_recall_pos",
        "test_f1_pos",
        "test_balanced_accuracy",
        "test_auroc",
        "test_auprc",
        "test_mcc",
        "tn",
        "fp",
        "fn",
        "tp"
    ]

    with open(all_runs_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow(r)

    summary_file = os.path.join(output_dir, "summary_mean_sd.csv")

    metric_cols = [
        "best_val_auprc",
        "val_mcc_at_threshold",
        "test_loss",
        "test_accuracy",
        "test_precision_pos",
        "test_recall_pos",
        "test_f1_pos",
        "test_balanced_accuracy",
        "test_auroc",
        "test_auprc",
        "test_mcc",
    ]

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "sd", "mean ± sd"])

        print("\n" + "=" * 90)
        print("Run summary: mean ± SD")
        print("=" * 90)

        for col in metric_cols:
            values = np.array([r[col] for r in results], dtype=float)
            mean_val = np.mean(values)
            sd_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            mean_sd_text = f"{mean_val:.4f} ± {sd_val:.4f}"

            writer.writerow([
                col,
                f"{mean_val:.6f}",
                f"{sd_val:.6f}",
                mean_sd_text
            ])

            print(f"{col}: {mean_sd_text}")

    print(f"\nAll run results saved to: {all_runs_file}")
    print(f"Summary mean ± SD saved to: {summary_file}")


if __name__ == "__main__":
    print("=" * 90)
    print("SecPPIS engineered local AA+SS feature MLP protein-level evaluation")
    print("Strategy: seed=42, engineered local features, original labels")
    print("=" * 90)

    metadata = build_protein_metadata(AA_DIR, SS_DIR, LABEL_DIR)

    all_results = []

    for seed in SEEDS:
        result = run_one_seed(metadata, seed)
        all_results.append(result)

    save_run_summary(all_results, BASE_OUTPUT_DIR)

    print("\nAll runs completed.")