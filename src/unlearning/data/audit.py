"""Decode, hash, and deduplicate the images named by a manifest."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image

from unlearning.splits import ManifestRow, sha256_file

VAL_SPLITS = {"val_tune", "val_test"}


@dataclass
class AuditResult:
    kept: list[ManifestRow] = field(default_factory=list)
    quarantine: list[dict] = field(default_factory=list)
    exact_duplicates: list[dict] = field(default_factory=list)
    near_duplicates: list[dict] = field(default_factory=list)


def audit_manifest(
    rows: list[ManifestRow], root: Path, phash_threshold: int = 4
) -> AuditResult:
    result = AuditResult()
    seen_hashes: dict[str, ManifestRow] = {}
    hashes: dict[str, tuple[str, str]] = {}

    for row in rows:
        path = root / row.relative_path
        if not path.is_file():
            result.quarantine.append({"path": row.relative_path, "reason": "missing"})
            continue
        if path.stat().st_size == 0:
            result.quarantine.append({"path": row.relative_path, "reason": "zero_byte"})
            continue

        try:
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError(f"non_rgb mode={image.mode}")
                image.load()
                phash = str(imagehash.phash(image))
        except (OSError, ValueError) as error:
            result.quarantine.append(
                {"path": row.relative_path, "reason": str(error)}
            )
            continue

        sha256 = sha256_file(path)
        if sha256 in seen_hashes:
            first = seen_hashes[sha256]
            is_val_copy = row.split in VAL_SPLITS and first.split not in VAL_SPLITS
            result.exact_duplicates.append(
                {
                    "dropped": row.relative_path,
                    "kept": first.relative_path,
                    "reason": (
                        "drop_val_duplicate_of_train"
                        if is_val_copy
                        else "drop_exact_duplicate"
                    ),
                }
            )
            continue

        seen_hashes[sha256] = row
        hashes[row.relative_path] = (sha256, phash)
        result.kept.append(row)

    by_class: dict[int, list[ManifestRow]] = defaultdict(list)
    for row in result.kept:
        by_class[row.class_index_0b].append(row)

    for group in by_class.values():
        for i, first in enumerate(group):
            for second in group[i + 1 :]:
                first_hash, first_phash = hashes[first.relative_path]
                second_hash, second_phash = hashes[second.relative_path]
                if first_hash == second_hash:
                    continue
                distance = imagehash.hex_to_hash(first_phash) - imagehash.hex_to_hash(
                    second_phash
                )
                if distance <= phash_threshold:
                    result.near_duplicates.append(
                        {
                            "first": first.relative_path,
                            "second": second.relative_path,
                            "phash_distance": distance,
                        }
                    )

    return result


def _write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not records:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_reports(audit: AuditResult, report_dir: Path) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "quarantine.csv", audit.quarantine)
    _write_csv(report_dir / "exact_duplicates.csv", audit.exact_duplicates)
    _write_csv(report_dir / "near_duplicates.csv", audit.near_duplicates)

    summary = {
        "kept": len(audit.kept),
        "quarantined": len(audit.quarantine),
        "exact_duplicates": len(audit.exact_duplicates),
        "near_duplicate_pairs": len(audit.near_duplicates),
    }
    (report_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
