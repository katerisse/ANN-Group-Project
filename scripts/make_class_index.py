"""Generate the committed 1000-class index from a pinned Keras JSON file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "imagenet_class_index.json"
)
MD5 = "c2c37ea517e94d9795004a39431a14cb"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "manifests" / "class_index.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = urllib.request.urlopen(URL, timeout=30).read()
    if hashlib.md5(payload).hexdigest() != MD5:
        raise SystemExit(f"MD5 mismatch for {URL}")

    rows = json.loads(payload)
    if len(rows) != 1000:
        raise SystemExit(f"expected 1000 classes, found {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["imagenet_index_1b", "model_index_0b", "synset", "name"])
        for index_0b in range(1000):
            synset, name = rows[str(index_0b)]
            writer.writerow([index_0b + 1, index_0b, synset, name])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
