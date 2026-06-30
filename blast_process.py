# -*- coding: utf-8 -*-
"""
BLAST preprocessing for NN benchmark -> save to one CSV:
    /home/chencanhui/Protein/Squidly-main/output/blast_nn.csv

What it does:
1) Read NN_enzyme_label.txt (3-line blocks: >id / seq / 0-1 labels)
2) Read reviewed_sprot_08042025.csv (Entry / Sequence / Residue)
3) Build DIAMOND database
4) Search each query against the database with DIAMOND BLASTP (--ultra-sensitive)
5) Keep the top hit (highest sequence identity, then bitscore)
6) Align top hit vs query using ClustalOmega
7) Transfer catalytic residues from reference to query through the gapped alignment
8) Save one CSV that can later be merged into the main Squidly inference by `id`

Assumptions:
- label=1 means catalytic residue
- DB column `Residue` is assumed 1-based by default and converted to 0-based for Python indexing
  If you later confirm your DB `Residue` is already 0-based, set RESIDUE_INDEX_BASE = 0
"""

import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from Bio import AlignIO


# =========================================================
# Fixed paths from your current setup
# =========================================================
DATASET_PATH = "/home/chencanhui/Protein/Dataset/HA_superfamily/HA_superfamily_enzyme_label.txt"
BLAST_DB_CSV = "/home/chencanhui/Protein/data/reviewed_sprot_08042025.csv/reviewed_sprot_08042025.csv"
OUTPUT_CSV = "/home/chencanhui/Protein/output/blast_HA_superfamily.csv"

# Work directory for fasta / DIAMOND DB / msa files
WORK_DIR = "/home/chencanhui/Protein/output/blast_HA_superfamily_work"

# =========================================================
# External tools
# =========================================================
DIAMOND_BIN = "/home/chencanhui/Protein/diamond/diamond"
CLUSTALO_BIN = "clustalo"

# =========================================================
# Config
# =========================================================
RESIDUE_INDEX_BASE = 0
DIAMOND_THREADS = 8
FORCE_REBUILD_DIAMOND_DB = False
KEEP_INTERMEDIATE_FILES = True

QUERY_FASTA = os.path.join(WORK_DIR, "HA_superfamily_queries.fasta")
DB_FASTA = os.path.join(WORK_DIR, "reviewed_sprot_db.fasta")
DB_PREFIX = os.path.join(WORK_DIR, "reviewed_sprot_db")
BLAST_TSV = os.path.join(WORK_DIR, "diamond_top_hits.tsv")
MSA_DIR = os.path.join(WORK_DIR, "msa")


# =========================================================
# Utilities
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def check_file_executable(path: str, name: str):
    if not os.path.exists(path):
        raise RuntimeError(f"{name} not found: {path}")
    if not os.access(path, os.X_OK):
        raise RuntimeError(f"{name} exists but is not executable: {path}")


def check_executable_in_path(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable not found in PATH: {name}\n"
            f"Please make sure `{name}` is installed and available in your shell PATH."
        )


def run_cmd(cmd: List[str], cwd: Optional[str] = None):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def sanitize_id(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "", s)
    return s


def parse_dataset_txt(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        lines = [x.strip() for x in f if x.strip()]

    if len(lines) % 3 != 0:
        raise RuntimeError(
            f"Dataset format error: non-empty line count {len(lines)} is not divisible by 3."
        )

    rows = []
    for i in range(0, len(lines), 3):
        header = lines[i]
        seq = lines[i + 1].strip()
        labels = lines[i + 2].strip()

        if not header.startswith(">"):
            raise RuntimeError(f"Expected FASTA-style header on line {i+1}, got: {header}")

        raw_id = header[1:].strip()

        if len(seq) != len(labels):
            raise RuntimeError(
                f"Label length mismatch for {raw_id}: len(seq)={len(seq)} vs len(labels)={len(labels)}"
            )

        if re.search(r"[^01]", labels):
            raise RuntimeError(
                f"Label string for {raw_id} contains non-binary chars: {labels[:100]}"
            )

        rows.append({
            "id": raw_id,
            "id_sanitized": sanitize_id(raw_id),
            "seq": seq,
            "labels": labels,
            "seq_len": len(seq),
        })

    df = pd.DataFrame(rows)

    # 如果 id、seq、labels 都完全一样，认为是重复拷贝，直接删除
    before = len(df)
    df = df.drop_duplicates(subset=["id_sanitized", "seq", "labels"], keep="first").copy()
    after = len(df)

    if before != after:
        print(f"[INFO] Removed exact duplicated query records: {before - after}")

    # 如果同一个 id_sanitized 还有多条，说明同名但内容不同，自动加后缀
    if df["id_sanitized"].duplicated().any():
        dup = df[df["id_sanitized"].duplicated(keep=False)]["id"].tolist()
        print(f"[WARN] Same id but different content found: {dup[:10]}")
        print("[WARN] Auto-renaming duplicated id_sanitized by adding _dupN suffix.")

        df["id_sanitized_base"] = df["id_sanitized"]
        dup_count = df.groupby("id_sanitized").cumcount()

        df["id_sanitized"] = [
            sid if n == 0 else f"{sid}_dup{n}"
            for sid, n in zip(df["id_sanitized"], dup_count)
        ]

    return df


def read_blast_db_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"Entry", "Sequence", "Residue"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"BLAST DB CSV missing columns: {sorted(missing)}")

    df = df[["Entry", "Sequence", "Residue"]].copy()
    df["Entry_raw"] = df["Entry"].astype(str).str.strip()
    df["Entry"] = df["Entry_raw"].map(sanitize_id)
    df["Sequence"] = df["Sequence"].astype(str).str.strip()
    df["Residue"] = df["Residue"].astype(str).str.strip()

    df = df[df["Entry"] != ""].copy()
    df = df[df["Sequence"] != ""].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["Entry"], keep="first").copy()
    after = len(df)
    if before != after:
        print(f"[INFO] Removed duplicated DB entries: {before - after}")

    return df


def parse_residue_field_to_0based(residue_str: str, residue_index_base: int = 1) -> List[int]:
    if residue_str is None:
        return []

    s = str(residue_str).strip()
    if s == "" or s.lower() == "nan":
        return []

    s = re.sub(r"[^0-9|]+", "", s)
    if s == "":
        return []

    out = []
    for token in s.split("|"):
        token = token.strip()
        if token == "":
            continue
        val = int(token)
        if residue_index_base == 1:
            val -= 1
        if val >= 0:
            out.append(val)

    return sorted(set(out))


def label_string_to_indices_0based(label_str: str) -> List[int]:
    return [i for i, c in enumerate(label_str) if c == "1"]


def indices_0based_to_label_string(length: int, idxs: List[int]) -> str:
    arr = ["0"] * length
    for x in idxs:
        if 0 <= x < length:
            arr[x] = "1"
    return "".join(arr)


def build_query_fasta(query_df: pd.DataFrame, fasta_path: str):
    with open(fasta_path, "w") as fout:
        for _, row in query_df.iterrows():
            fout.write(f">{row['id_sanitized']}\n{row['seq']}\n")


def build_db_fasta(db_df: pd.DataFrame, fasta_path: str):
    with open(fasta_path, "w") as fout:
        for _, row in db_df.iterrows():
            fout.write(f">{row['Entry']}\n{row['Sequence']}\n")


def build_diamond_db(db_fasta: str, db_prefix: str, force: bool = False):
    dmnd_file = db_prefix + ".dmnd"
    if (not force) and os.path.exists(dmnd_file):
        print(f"[INFO] DIAMOND DB already exists, skip build: {dmnd_file}")
        return

    run_cmd([
        DIAMOND_BIN, "makedb",
        "--in", db_fasta,
        "-d", db_prefix
    ])


def run_diamond_top_hit(query_fasta: str, db_prefix: str, out_tsv: str, threads: int = 8):
    run_cmd([
        DIAMOND_BIN, "blastp",
        "--query", query_fasta,
        "--db", db_prefix,
        "--out", out_tsv,
        "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue", "bitscore",
        "--ultra-sensitive",
        "--max-target-seqs", "1",
        "--threads", str(threads),
    ])


def read_top_hit_tsv(tsv_path: str) -> pd.DataFrame:
    if (not os.path.exists(tsv_path)) or os.path.getsize(tsv_path) == 0:
        return pd.DataFrame(columns=["id_sanitized", "blast_target", "blast_identity", "blast_evalue", "blast_bitscore"])

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["id_sanitized", "blast_target", "blast_identity", "blast_evalue", "blast_bitscore"]
    )

    df = df.sort_values(
        by=["blast_identity", "blast_bitscore"],
        ascending=[False, False]
    ).copy()

    df = df.drop_duplicates(subset=["id_sanitized"], keep="first").copy()
    return df


def write_pair_fasta(ref_id: str, ref_seq: str, query_id: str, query_seq: str, out_fasta: str):
    with open(out_fasta, "w") as fout:
        fout.write(f">{ref_id}\n{ref_seq}\n")
        fout.write(f">{query_id}\n{query_seq}\n")


def run_clustalo_pairwise(in_fa: str, out_msa: str):
    run_cmd([
        CLUSTALO_BIN,
        "--force",
        "-i", in_fa,
        "-o", out_msa
    ])


def map_ref_residues_to_query_via_alignment(
    msa_path: str,
    ref_id: str,
    query_id: str,
    ref_active_idx_0based: List[int],
) -> List[int]:
    alignment = AlignIO.read(msa_path, "fasta")
    records = {rec.id: str(rec.seq) for rec in alignment}

    if ref_id not in records or query_id not in records:
        raise RuntimeError(
            f"Alignment file missing ids. Need ref={ref_id}, query={query_id}, file={msa_path}"
        )

    ref_aln = records[ref_id]
    query_aln = records[query_id]

    if len(ref_aln) != len(query_aln):
        raise RuntimeError(f"Alignment length mismatch in {msa_path}")

    active_set = set(ref_active_idx_0based)

    ref_pos = 0
    query_pos = 0
    mapped = []

    for i in range(len(ref_aln)):
        r = ref_aln[i]
        q = query_aln[i]

        current_ref_pos = ref_pos if r != "-" else None
        current_query_pos = query_pos if q != "-" else None

        if current_ref_pos is not None and current_ref_pos in active_set:
            if current_query_pos is not None:
                mapped.append(current_query_pos)

        if r != "-":
            ref_pos += 1
        if q != "-":
            query_pos += 1

    return sorted(set(mapped))


def compute_binary_metrics(true_label_string: str, pred_label_string: str) -> Dict[str, float]:
    if len(true_label_string) != len(pred_label_string):
        raise RuntimeError(
            f"Metric error: len(true)={len(true_label_string)} != len(pred)={len(pred_label_string)}"
        )

    tp = fp = fn = tn = 0
    for t, p in zip(true_label_string, pred_label_string):
        if t == "1" and p == "1":
            tp += 1
        elif t == "0" and p == "1":
            fp += 1
        elif t == "1" and p == "0":
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# =========================================================
# Main
# =========================================================
def main():
    ensure_dir(os.path.dirname(OUTPUT_CSV))
    ensure_dir(WORK_DIR)
    ensure_dir(MSA_DIR)

    check_file_executable(DIAMOND_BIN, "DIAMOND_BIN")
    check_executable_in_path(CLUSTALO_BIN)

    print("[INFO] Reading NN dataset...")
    query_df = parse_dataset_txt(DATASET_PATH)
    print(f"[INFO] Query count: {len(query_df)}")

    print("[INFO] Reading BLAST DB CSV...")
    db_df = read_blast_db_csv(BLAST_DB_CSV)
    print(f"[INFO] DB entry count: {len(db_df)}")

    db_df["db_residue_idx_0based"] = db_df["Residue"].map(
        lambda x: parse_residue_field_to_0based(x, residue_index_base=RESIDUE_INDEX_BASE)
    )

    entry_to_seq: Dict[str, str] = dict(zip(db_df["Entry"], db_df["Sequence"]))
    entry_to_raw: Dict[str, str] = dict(zip(db_df["Entry"], db_df["Entry_raw"]))
    entry_to_residues: Dict[str, List[int]] = dict(zip(db_df["Entry"], db_df["db_residue_idx_0based"]))

    print("[INFO] Writing query FASTA...")
    build_query_fasta(query_df, QUERY_FASTA)

    print("[INFO] Writing DB FASTA...")
    build_db_fasta(db_df, DB_FASTA)

    print("[INFO] Building / checking DIAMOND DB...")
    build_diamond_db(DB_FASTA, DB_PREFIX, force=FORCE_REBUILD_DIAMOND_DB)

    print("[INFO] Running DIAMOND top-hit search...")
    run_diamond_top_hit(QUERY_FASTA, DB_PREFIX, BLAST_TSV, threads=DIAMOND_THREADS)

    print("[INFO] Reading BLAST top-hit results...")
    hit_df = read_top_hit_tsv(BLAST_TSV)
    print(f"[INFO] Queries with at least one BLAST hit: {len(hit_df)} / {len(query_df)}")

    merged = query_df.merge(hit_df, on="id_sanitized", how="left")

    merged["blast_target_raw"] = None
    merged["has_blast_hit"] = merged["blast_target"].notna()
    merged["alignment_ok"] = False
    merged["mapping_note"] = ""
    merged["db_target_residue_idx_0based"] = ""
    merged["BLAST_residue_idx_0based"] = ""
    merged["BLAST_label_string"] = None
    merged["mapped_residue_count"] = 0
    merged["true_residue_idx_0based"] = merged["labels"].map(
        lambda s: "|".join(str(x) for x in label_string_to_indices_0based(s))
    )

    print("[INFO] Running pairwise ClustalOmega + residue transfer...")
    for idx, row in tqdm(merged.iterrows(), total=len(merged), dynamic_ncols=True):
        qid = row["id_sanitized"]
        qseq = row["seq"]
        qlen = row["seq_len"]
        target = row["blast_target"]

        pred_label_string = "0" * qlen
        pred_idx: List[int] = []

        if pd.isna(target) or not isinstance(target, str) or target == "":
            merged.at[idx, "has_blast_hit"] = False
            merged.at[idx, "mapping_note"] = "no_blast_hit"
            merged.at[idx, "BLAST_label_string"] = pred_label_string
            continue

        if target not in entry_to_seq:
            merged.at[idx, "mapping_note"] = "target_missing_in_db_lookup"
            merged.at[idx, "BLAST_label_string"] = pred_label_string
            continue

        ref_seq = entry_to_seq[target]
        ref_residue_idx = entry_to_residues.get(target, [])
        merged.at[idx, "blast_target_raw"] = entry_to_raw.get(target, target)
        merged.at[idx, "db_target_residue_idx_0based"] = "|".join(str(x) for x in ref_residue_idx)

        if len(ref_residue_idx) == 0:
            merged.at[idx, "mapping_note"] = "target_has_no_catalytic_annotation"
            merged.at[idx, "BLAST_label_string"] = pred_label_string
            continue

        pair_fa = os.path.join(MSA_DIR, f"{target}__{qid}.fa")
        pair_msa = os.path.join(MSA_DIR, f"{target}__{qid}.msa")

        write_pair_fasta(target, ref_seq, qid, qseq, pair_fa)

        try:
            run_clustalo_pairwise(pair_fa, pair_msa)
            mapped_idx = map_ref_residues_to_query_via_alignment(
                msa_path=pair_msa,
                ref_id=target,
                query_id=qid,
                ref_active_idx_0based=ref_residue_idx,
            )
            pred_idx = mapped_idx
            pred_label_string = indices_0based_to_label_string(qlen, pred_idx)

            merged.at[idx, "alignment_ok"] = True
            merged.at[idx, "mapping_note"] = "ok"
        except Exception as e:
            merged.at[idx, "alignment_ok"] = False
            merged.at[idx, "mapping_note"] = f"alignment_or_mapping_failed: {str(e)}"
            pred_idx = []
            pred_label_string = "0" * qlen

        merged.at[idx, "BLAST_residue_idx_0based"] = "|".join(str(x) for x in pred_idx)
        merged.at[idx, "BLAST_label_string"] = pred_label_string
        merged.at[idx, "mapped_residue_count"] = len(pred_idx)

    metric_rows = []
    for _, row in merged.iterrows():
        m = compute_binary_metrics(row["labels"], row["BLAST_label_string"])
        metric_rows.append(m)

    metric_df = pd.DataFrame(metric_rows)
    merged = pd.concat([merged.reset_index(drop=True), metric_df.reset_index(drop=True)], axis=1)

    out_cols = [
        "id",
        "id_sanitized",
        "seq",
        "seq_len",
        "labels",
        "true_residue_idx_0based",
        "has_blast_hit",
        "blast_target",
        "blast_target_raw",
        "blast_identity",
        "blast_evalue",
        "blast_bitscore",
        "alignment_ok",
        "mapping_note",
        "db_target_residue_idx_0based",
        "BLAST_residue_idx_0based",
        "BLAST_label_string",
        "mapped_residue_count",
        "tp", "fp", "fn", "tn",
        "precision", "recall", "f1", "accuracy",
    ]
    merged = merged[out_cols].copy()

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved BLAST results to: {OUTPUT_CSV}")

    total_tp = int(merged["tp"].sum())
    total_fp = int(merged["fp"].sum())
    total_fn = int(merged["fn"].sum())
    total_tn = int(merged["tn"].sum())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0 else 0.0
    )
    overall_accuracy = (
        (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0
    )

    hit_queries = int(merged["has_blast_hit"].sum())
    aligned_queries = int(merged["alignment_ok"].sum())
    mapped_queries = int((merged["mapped_residue_count"] > 0).sum())

    print("\n===== BLAST PREPROCESS SUMMARY =====")
    print("Queries total           :", len(merged))
    print("Queries with BLAST hit  :", hit_queries)
    print("Queries aligned         :", aligned_queries)
    print("Queries with mapped AS  :", mapped_queries)
    print("TP                      :", total_tp)
    print("FP                      :", total_fp)
    print("FN                      :", total_fn)
    print("TN                      :", total_tn)
    print("Precision               :", overall_precision)
    print("Recall                  :", overall_recall)
    print("F1                      :", overall_f1)
    print("Accuracy                :", overall_accuracy)

    if not KEEP_INTERMEDIATE_FILES:
        if os.path.exists(MSA_DIR):
            shutil.rmtree(MSA_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
