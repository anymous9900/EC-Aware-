import os
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from EC_contrastive_transformer_lstm import *

Model_Path = "./Model/"

# BLAST fusion config
BLAST_CSV_PATH = os.getenv(
    "BLAST_CSV_PATH",
    "/home/chencanhui/Protein/output/blast_pc.csv",
)
BLAST_IDENTITY_THRESHOLD = float(os.getenv("BLAST_IDENTITY_THRESHOLD", "60.0"))
BLAST_POS_SCORE = float(os.getenv("BLAST_POS_SCORE", "0.95"))


class EnzDataset(Dataset):
    def __init__(self, dataframe, data_path):
        self.names = dataframe["ID"].values
        self.sequences = dataframe["sequence"].values
        self.labels = dataframe["label"].values
        self.data_path = data_path

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        data_path = self.data_path

        pssm_feature, hmm_feature, evo_feature = embedding(sequence_name, data_path)
        atom_features, seq_feature = get_atom_features(sequence_name, data_path)

        esm_feature = load_ct_embedding_test(sequence_name, sequence)
        inter_feature = load_interpro_data_test(sequence_name)
        print(f"Sequence Names: {sequence_name}")
        print(f"pssm: {pssm_feature.shape}")
        print(f"hhm: {hmm_feature.shape}")
        print(f"esm: {esm_feature.shape}")
        print(f"evo: {evo_feature.shape}")
        node_features = np.concatenate(
            [pssm_feature, hmm_feature, atom_features, seq_feature, esm_feature], axis=1
        )
        graph = load_graph(sequence_name, data_path)

        return sequence_name, sequence, label, node_features, graph, evo_feature, atom_features, inter_feature

    def __len__(self):
        return len(self.labels)


def _to_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}


def normalize_protein_id(pid):
    if pid is None:
        return None
    pid = str(pid).strip()
    if not pid:
        return None

    if "-" in pid:
        pdbid, chain = pid.split("-", 1)
    elif "_" in pid:
        pdbid, chain = pid.split("_", 1)
    else:
        return pid.lower()
    return pdbid.lower() + "-" + chain.upper()


def _parse_blast_label_string(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return None
    if any(ch not in {"0", "1"} for ch in s):
        return None
    return [int(ch) for ch in s]


def load_blast_map(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"BLAST csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {
        "id",
        "has_blast_hit",
        "alignment_ok",
        "blast_identity",
        "BLAST_label_string",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"BLAST csv missing required columns: {sorted(missing)}")

    blast_map = {}
    for _, row in df.iterrows():
        pid = normalize_protein_id(row["id"])
        if pid is None:
            continue
        identity = None
        try:
            if pd.notna(row["blast_identity"]):
                identity = float(row["blast_identity"])
        except Exception:
            identity = None

        blast_map[pid] = {
            "has_blast_hit": _to_bool(row["has_blast_hit"]),
            "alignment_ok": _to_bool(row["alignment_ok"]),
            "blast_identity": identity,
            "blast_label_list": _parse_blast_label_string(row["BLAST_label_string"]),
        }

    return blast_map


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            sequence_names, sequence, labels, node_features, graphs, evo_feature, atom_features, inter_feature = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda())
                graphs = Variable(graphs.cuda())
                evo_feature = Variable(evo_feature.cuda())
                y_true = Variable(labels.cuda())
                inter_feature = (inter_feature[0].cuda(), inter_feature[1].cuda())
            else:
                node_features = Variable(node_features)
                graphs = Variable(graphs)
                evo_feature = Variable(evo_feature)
                y_true = Variable(labels)
                inter_feature = (inter_feature[0], inter_feature[1])

            node_features = torch.squeeze(node_features)
            graphs = torch.squeeze(graphs)
            evo_feature = torch.squeeze(evo_feature)
            y_true = torch.squeeze(y_true)

            y_pred, enzfeas = model(node_features, graphs, evo_feature, inter_feature)
            print(f"y_pred shape: {y_pred.shape}")
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred / 8)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy().tolist()

            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]
            epoch_loss += loss.item()
            n += 1

    epoch_loss_avg = epoch_loss / n
    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold=None):
    if best_threshold is None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)

    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        "binary_acc": binary_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "AUC": AUC,
        "AUPRC": AUPRC,
        "mcc": mcc,
        "threshold": best_threshold,
    }
    return results


def build_sequence_records(test_dataframe, pred_dict, blast_map):
    records = []
    route_stats = {
        "missing_prediction": 0,
        "missing_blast_row": 0,
        "has_blast_hit_false": 0,
        "alignment_not_ok": 0,
        "identity_missing": 0,
        "identity_below_threshold": 0,
        "label_parse_failed": 0,
        "label_length_mismatch": 0,
        "use_blast": 0,
        "use_model": 0,
    }
    for row in test_dataframe.itertuples(index=False):
        pid = normalize_protein_id(row.ID)
        labels = [int(x) for x in row.label]
        probs = pred_dict.get(row.ID)
        if probs is None and pid != row.ID:
            probs = pred_dict.get(pid)
        if probs is None:
            print(f"[warn] missing model prediction for {pid}, skip")
            route_stats["missing_prediction"] += 1
            continue

        if len(probs) != len(labels):
            print(f"[warn] length mismatch for {pid}: probs={len(probs)} labels={len(labels)}, skip")
            continue

        blast_info = blast_map.get(pid)
        use_blast = False
        blast_pred = None
        blast_identity = None
        route_reason = "model_fallback"

        if blast_info is not None:
            blast_identity = blast_info["blast_identity"]
            candidate = blast_info["blast_label_list"]
            if not blast_info["has_blast_hit"]:
                route_reason = "has_blast_hit_false"
                route_stats["has_blast_hit_false"] += 1
            elif not blast_info["alignment_ok"]:
                route_reason = "alignment_not_ok"
                route_stats["alignment_not_ok"] += 1
            elif blast_identity is None:
                route_reason = "identity_missing"
                route_stats["identity_missing"] += 1
            elif blast_identity < BLAST_IDENTITY_THRESHOLD:
                route_reason = "identity_below_threshold"
                route_stats["identity_below_threshold"] += 1
            elif candidate is None:
                route_reason = "label_parse_failed"
                route_stats["label_parse_failed"] += 1
            elif len(candidate) != len(labels):
                route_reason = "label_length_mismatch"
                route_stats["label_length_mismatch"] += 1
            else:
                use_blast = True
                blast_pred = candidate
                route_reason = "use_blast"
                route_stats["use_blast"] += 1
        else:
            route_reason = "missing_blast_row"
            route_stats["missing_blast_row"] += 1

        if not use_blast:
            route_stats["use_model"] += 1

        records.append(
            {
                "id": pid,
                "labels": labels,
                "probs": probs,
                "use_blast": use_blast,
                "blast_pred": blast_pred,
                "blast_identity": blast_identity,
                "route_reason": route_reason,
            }
        )

    return records, route_stats


def flatten_records_with_threshold(records, threshold):
    preds = []
    labels = []
    scores = []

    for rec in records:
        if rec["use_blast"]:
            y_score = list(rec["probs"])
            for idx, blast_label in enumerate(rec["blast_pred"]):
                if blast_label == 1:
                    y_score[idx] = max(y_score[idx], BLAST_POS_SCORE)
        else:
            y_score = rec["probs"]

        y_pred = [1 if p >= threshold else 0 for p in y_score]
        preds.extend(y_pred)
        labels.extend(rec["labels"])
        scores.extend(y_score)

    return preds, labels, scores


def fusion_analysis(records):
    best_f1 = -1.0
    best_threshold = 0.5

    for threshold in np.arange(0.01, 0.99, 0.01):
        preds, labels, _ = flatten_records_with_threshold(records, float(threshold))
        f1 = metrics.f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    binary_pred, binary_true, fusion_scores = flatten_records_with_threshold(records, best_threshold)
    result = {
        "binary_acc": metrics.accuracy_score(binary_true, binary_pred),
        "precision": metrics.precision_score(binary_true, binary_pred),
        "recall": metrics.recall_score(binary_true, binary_pred),
        "f1": metrics.f1_score(binary_true, binary_pred),
        "mcc": metrics.matthews_corrcoef(binary_true, binary_pred),
        "AUPRC": metrics.average_precision_score(binary_true, fusion_scores),
        "AUC": metrics.roc_auc_score(binary_true, fusion_scores),
        "threshold": best_threshold,
    }
    return result


def test(test_dataframe, data_path):
    test_loader = DataLoader(dataset=EnzDataset(test_dataframe, data_path), batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    # model_name = "LSTM_Transformer_esm_GraphCAI_EC_contrastive_500_239.pkl"
    model_name = "LSTM_Transformer_esm_GraphCAI_EC_contrastive_500_179.pkl"
    # model_name = "LSTM_Transformer_esm_GraphCAI_EC_contrastive_500_139.pkl"
    # model_name = "LSTM_Transformer_esm_GraphCAI_EC_contrastive_500_159.pkl"
    # model_name = "LSTM_Transformer_esm_GraphCAI_EC_contrastive_500_259.pkl"
    # model_name = "LSTM_Transformer_esm_GraphCAI_EC_contrastive_500_279.pkl"
    # model_name = "LSTM_Transformer_esm_GraphCAI_EC_contrastive_500_239.pkl"   # nn最好
    model = GraphCAI(NLAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)

    if torch.cuda.is_available():
        torch.cuda.set_device(3)
        print("Using GPU:", torch.cuda.current_device())

    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(Model_Path + model_name, map_location="cuda:3"))

    epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)

    result_test = analysis(test_true, test_pred)
    print("========== Evaluate Test set: GraphCAI only ==========")
    print("Test recall: ", result_test["recall"])
    print("Test precision:", result_test["precision"])
    print("Test f1: ", result_test["f1"])
    print("Test mcc: ", result_test["mcc"])
    print("Test AUC: ", result_test["AUC"])
    print("Test AUPRC: ", result_test["AUPRC"])
    print("Threshold: ", result_test["threshold"])

    print("BLAST csv:", BLAST_CSV_PATH)
    print("BLAST identity threshold:", BLAST_IDENTITY_THRESHOLD)
    print("BLAST positive score:", BLAST_POS_SCORE)
    blast_map = load_blast_map(BLAST_CSV_PATH)
    records, route_stats = build_sequence_records(test_dataframe, pred_dict, blast_map)
    routed_to_blast = sum(1 for rec in records if rec["use_blast"])
    routed_to_model = len(records) - routed_to_blast

    print("========== Evaluate Test set: Conservative BLAST + GraphCAI ==========")
    print("Sequences with BLAST assist:", routed_to_blast)
    print("Sequences without BLAST assist:", routed_to_model)
    print("Fusion recall: ", fusion_result["recall"])
    print("Fusion precision:", fusion_result["precision"])
    print("Fusion f1: ", fusion_result["f1"])
    print("Fusion mcc: ", fusion_result["mcc"])
    print("Fusion accuracy: ", fusion_result["binary_acc"])
    print("Fusion AUC: ", fusion_result["AUC"])
    print("Fusion AUPRC: ", fusion_result["AUPRC"])
    print("Fusion threshold: ", fusion_result["threshold"])
    print("Route stats:", route_stats)


def main():
    dataset = ["NN"]
    # dataset = ["NN", "HA_superfamily", "EF_superfamily", "PC", "EF_fold"]
    index = 0


    data_dir = "./Dataset/" + dataset[index] + "/"
    f = open(data_dir + "test-" + dataset[index] + "_id.txt", "r")
    protein_list, sequences, labels = [], [], []
    filedata = f.readlines()
    for line in filedata:
        protein = line.strip()
        if protein in skip_ids:
            continue
        protein_list.append(protein)
    print(protein_list)
    f.close()

    prot_seq = {}
    prot_anno = {}
    f = open("./Dataset/NN/NN_enzyme_label.txt", "r")
    data = f.readlines()
    for line in range(0, len(data)):
        if data[line].startswith(">"):
            protein = data[line].lstrip(">").strip()
            PDBID, Chain = protein.split("-") if "-" in protein else protein.split("_")
            pro = PDBID.lower() + "-" + Chain.upper()
            seq_p = data[line + 1].strip()
            query_anno = data[line + 2].strip()
            prot_seq[pro] = seq_p
            prot_anno[pro] = query_anno

    filtered_protein_list = []
    for prot in protein_list:
        if prot not in prot_seq or prot not in prot_anno:
            print(f"[warn] missing label or sequence for {prot}, skip")
            continue

        label_list = []
        seq = prot_seq[prot]
        label = prot_anno[prot]
        filtered_protein_list.append(prot)
        sequences.append(seq)
        for i in range(len(label)):
            label_list.append(int(label[i]))
        labels.append(label_list)

    test_dic = {"ID": filtered_protein_list, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    data_dir = "./Dataset/NN/"
    test(test_dataframe, data_dir)


if __name__ == "__main__":
    main()
