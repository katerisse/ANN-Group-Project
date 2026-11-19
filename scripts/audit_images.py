"""Audit manifest images and write quarantine/duplicate reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unlearning.classes import (
    check_synset_mapping,
    default_class_index_path,
    load_class_index,
)
from unlearning.config import DataConfig
from unlearning.data.audit import audit_manifest, write_reports
from unlearning.data.paths import get_root
from unlearning.data.verify import load_synset_mapping_from_devkit
from unlearning.splits import load_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=default_class_index_path())
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--devkit", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = get_root(args.root)
    index = load_class_index(args.index)
    if args.devkit is not None:
        synsets = load_synset_mapping_from_devkit(args.devkit)
        differences = check_synset_mapping(index, synsets)
        if differences:
            raise SystemExit(f"devkit synset mismatch: {differences}")

    rows = load_manifest(args.manifest)
    audit = audit_manifest(rows, root)
    summary = write_reports(audit, args.report_dir)
    print(summary)

    if args.apply:
        meta_path = args.manifest.with_suffix(".meta.json")
        seed = 0
        if meta_path.is_file():
            seed = json.loads(meta_path.read_text()).get("seed", 0)
        target = args.manifest.with_name("phase1_splits.audited.csv")
        digest = write_manifest(audit.kept, target, args.index, DataConfig(seed=seed))
        print(f"wrote {target} with sha256 {digest}")


if __name__ == "__main__":
    main()
