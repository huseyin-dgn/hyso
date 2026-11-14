"""Bounded offline UCI digits integration; never fetches data or installs packages.

The installed sklearn collection is the original UCI TEST partition. These new
splits share that collection and do not establish writer-independent evaluation.
Run explicitly: python -m examples.models.real_digits_acceptance --output QA_PATH
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hyso.datasets import load_image_splits
from .image_classification import run


SOURCE = "uci:10.24432/C50P49:optdigits.tes:sklearn.load_digits"
UCI_URL = "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"
SKLEARN_URL = "https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_digits.html"
CONVERSION = dict(
    input_range=[0, 16], output_range=[0, 255], mode="L", size=[8, 8],
    formula="uint8((uint16(source_pixel) * 255 + 8) // 16)",
    rounding="nearest, ties up", png_compress_level=9, png_optimize=False,
    training_tensor="PNG integer / 255, CHW fp32; no normalization",
)


def png_bytes(image):
    pixels = ((image.astype(np.uint16) * 255 + 8) // 16).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=9, optimize=False)
    return buffer.getvalue()


def prepare(root, *, seed=1358):
    """Convert exactly 12/4/4 per class into a fresh explicitly chosen directory.

    No sklearn fetch_* API is used. Source identities are zero-based row indices
    in the bundled optdigits.tes copy, not invented writer identifiers.
    """
    import sklearn
    from sklearn.datasets import load_digits

    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("Use a fresh digits directory")
    digits = load_digits()
    if (digits.images.shape != (1797, 8, 8) or digits.target.shape != (1797,)
            or not np.isfinite(digits.images).all()
            or not np.equal(digits.images, digits.images.astype(np.uint8)).all()
            or digits.images.min() != 0 or digits.images.max() != 16):
        raise ValueError("Unexpected bundled digits source")
    rng = np.random.default_rng(seed)
    records, used_content, duplicates = [], set(), []
    # One fixed per-class permutation; duplicate filtering precedes split assignment.
    for label in range(10):
        candidates = rng.permutation(np.flatnonzero(digits.target == label))
        chosen = []
        for index in candidates:
            original = digits.images[index].astype(np.uint8).tobytes()
            identity = hashlib.sha256(original).hexdigest()
            if identity in used_content:
                duplicates.append(int(index))
                continue
            used_content.add(identity)
            chosen.append(int(index))
            if len(chosen) == 20:
                break
        if len(chosen) != 20:
            raise ValueError("Insufficient distinct source images")
        for position, index in enumerate(chosen):
            split = "train" if position < 12 else "val" if position < 16 else "test"
            relative = f"{split}/{label}/source-{index:04d}.png"
            blob = png_bytes(digits.images[index])
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(blob)
            records.append(dict(
                source_index=index, label=label, source_id=f"{SOURCE}:row:{index}",
                original_source_id="optdigits.tes", source_row_zero_based=index,
                original_pixels_sha256=hashlib.sha256(
                    digits.images[index].astype(np.uint8).tobytes()
                ).hexdigest(),
                conversion=CONVERSION, path=relative, split=split,
                png_sha256=hashlib.sha256(blob).hexdigest(),
            ))
    data = load_image_splits(root, channels=1, image_size=(8, 8))
    manifest = data.to_dict()
    by_path = {row["path"]: row for row in manifest["samples"]}
    for row in records:
        row["tensor_sha256"] = by_path[row["path"]]["tensor_sha256"]
    return dict(
        source=SOURCE, source_url=UCI_URL, sklearn_url=SKLEARN_URL,
        citation="Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI. DOI:10.24432/C50P49",
        license="CC BY 4.0 (UCI source record verified 2026-11-14)",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="UCI processed 8x8 block counts converted deterministically to PNG; new subset splits",
        sklearn_version=sklearn.__version__, numpy_version=np.__version__,
        source_array_sha256=hashlib.sha256(
            digits.images.astype(np.uint8).tobytes() + digits.target.astype("<i8").tobytes()
        ).hexdigest(),
        real_world_source=True, raw_original_handwriting=False,
        original_uci_partition="test", writer_independence=False,
        prior_use="No load_digits use found in repository examples/tests/docs; external prior use unknown",
        seed=seed, budget=dict(train=120, val=40, test=40),
        duplicate_candidates_skipped=duplicates, conversion=CONVERSION,
        dataset_id=manifest["dataset_id"], samples=records,
    )


def main(output):
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or output.is_relative_to(repo):
        raise ValueError("Acceptance output must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    provenance = prepare(output / "digits")
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False), encoding="utf8"
    )
    # Configuration/budgets are fixed before either test evaluation. No selection.
    for kind in ("builder", "graph"):
        result = run(
            output / "digits", output / kind, model_kind=kind,
            channels=1, image_size=(8, 8), epochs=2, batch_size=16, seed=1358,
            dataset_origin="caller_supplied", dataset_provenance={
                key: value for key, value in provenance.items() if key != "samples"
            },
        )
        print(json.dumps(dict(
            model=kind, dataset_id=result["dataset_id"],
            accuracy=result["evaluation"]["metrics"]["accuracy"],
            native_checkpoint=result["native_checkpoint_binding"]["status"],
        )))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
