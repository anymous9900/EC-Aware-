import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm


DEFAULT_ESM_DIR = "/home/sharedata/chencanhui/esm15B/HA_superfamily"
DEFAULT_CONTRASTIVE_PATH = (
    "/home/chencanhui/Protein/"
    "train_out_trial/models/uniprot_contrastive_best.pt"
)
DEFAULT_REPR_LAYER = 48
DEFAULT_OUTPUT_KEY = "contrastive_representations"


class ContrastiveModel(nn.Module):
    def __init__(self, input_dim: int = 5120, output_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2560)
        self.fc2 = nn.Linear(2560, 1280)
        self.fc3 = nn.Linear(1280, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project saved ESM-15B residue representations from 5120 dims to 128 dims "
            "with a trained contrastive model, then write the projected features back "
            "into each .pt file."
        )
    )
    parser.add_argument(
        "--esm-dir",
        default=DEFAULT_ESM_DIR,
        help="Directory containing saved ESM .pt feature files.",
    )
    parser.add_argument(
        "--contrastive-path",
        default=DEFAULT_CONTRASTIVE_PATH,
        help="Path to the trained contrastive model checkpoint.",
    )
    parser.add_argument(
        "--repr-layer",
        type=int,
        default=DEFAULT_REPR_LAYER,
        help="ESM representation layer key to read from payload['representations'].",
    )
    parser.add_argument(
        "--output-key",
        default=DEFAULT_OUTPUT_KEY,
        help="Key used to store the projected [seq_len, 128] tensor.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for projection, for example 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output key if it already exists in a file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count files without writing changes.",
    )
    return parser.parse_args()


def load_contrastive_model(checkpoint_path: Path, device: torch.device) -> ContrastiveModel:
    model = ContrastiveModel()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch.float32).eval()
    return model


def iter_target_files(esm_dir: Path) -> Iterable[Path]:
    for path in sorted(esm_dir.glob("*.pt")):
        if path.name and path.name[0].isdigit():
            yield path


def extract_representation(payload: Dict, repr_layer: int) -> torch.Tensor:
    representations = payload.get("representations")
    if not isinstance(representations, dict):
        raise KeyError("missing dict field: 'representations'")

    if repr_layer not in representations:
        available = sorted(representations.keys())
        raise KeyError(f"layer {repr_layer} not found in 'representations'; available={available}")

    rep = representations[repr_layer]
    if not torch.is_tensor(rep):
        raise TypeError(f"representations[{repr_layer}] is not a torch.Tensor")
    if rep.ndim != 2:
        raise ValueError(f"representations[{repr_layer}] must be 2D, got shape={tuple(rep.shape)}")
    if rep.shape[-1] != 5120:
        raise ValueError(
            f"representations[{repr_layer}] last dim must be 5120, got shape={tuple(rep.shape)}"
        )
    return rep


def project_representation(
    model: ContrastiveModel,
    rep: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        projected = model(rep.to(device=device, dtype=torch.float32))
    return projected.cpu()


def process_file(
    pt_path: Path,
    model: ContrastiveModel,
    device: torch.device,
    repr_layer: int,
    output_key: str,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[bool, str]:
    payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, dict):
        return False, "payload is not a dict"

    if output_key in payload and not overwrite:
        return False, f"skip: '{output_key}' already exists"

    rep = extract_representation(payload, repr_layer)
    projected = project_representation(model, rep, device)

    if projected.shape[0] != rep.shape[0] or projected.shape[1] != 128:
        return False, f"unexpected projected shape={tuple(projected.shape)}"

    if dry_run:
        return True, f"dry-run ok: {tuple(rep.shape)} -> {tuple(projected.shape)}"

    payload[output_key] = {
        repr_layer: projected,
    }
    torch.save(payload, pt_path)
    return True, f"saved: {tuple(rep.shape)} -> {tuple(projected.shape)}"


def main() -> None:
    args = parse_args()

    esm_dir = Path(args.esm_dir)
    contrastive_path = Path(args.contrastive_path)
    device = torch.device(args.device)

    if not esm_dir.exists():
        raise FileNotFoundError(f"esm dir not found: {esm_dir}")
    if not contrastive_path.exists():
        raise FileNotFoundError(f"contrastive checkpoint not found: {contrastive_path}")

    files = list(iter_target_files(esm_dir))
    if not files:
        raise RuntimeError(f"no digit-prefixed .pt files found under: {esm_dir}")

    model = load_contrastive_model(contrastive_path, device)

    processed = 0
    skipped = 0
    failed = 0

    for pt_path in tqdm(files, desc="project contrastive", dynamic_ncols=True):
        try:
            ok, message = process_file(
                pt_path=pt_path,
                model=model,
                device=device,
                repr_layer=args.repr_layer,
                output_key=args.output_key,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            if ok:
                processed += 1
            else:
                skipped += 1
            print(f"[{pt_path.name}] {message}")
        except Exception as exc:
            failed += 1
            print(f"[{pt_path.name}] failed: {exc}")

    print("=" * 80)
    print(f"esm_dir={esm_dir}")
    print(f"contrastive_path={contrastive_path}")
    print(f"device={device}")
    print(f"repr_layer={args.repr_layer}")
    print(f"output_key={args.output_key}")
    print(f"files_total={len(files)}")
    print(f"processed={processed}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    print(f"dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
