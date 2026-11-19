import hashlib

import pytest

import download_imagenet
from unlearning.data import verify


def _write_archives(raw_dir):
    names = ("ILSVRC2012_img_val.tar", "ILSVRC2012_devkit_t12.tar.gz")
    for name in names:
        path = raw_dir / name
        path.write_bytes(name.encode())
    return names


def test_verify_archives_accepts_matching_checksums(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    names = _write_archives(raw_dir)
    for name in names:
        digest = hashlib.md5((raw_dir / name).read_bytes()).hexdigest()
        monkeypatch.setitem(verify.ARCHIVE_MD5, name, digest)

    report = verify.verify_archives(raw_dir)

    assert report.ok
    assert set(report.checked) == set(names)


def test_verify_archives_fails_closed_on_mismatch(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    names = _write_archives(raw_dir)
    for name in names:
        monkeypatch.setitem(verify.ARCHIVE_MD5, name, "0" * 32)

    report = verify.verify_archives(raw_dir)

    assert not report.ok
    assert len(report.errors) == 2


def test_gdrive_adapter_uses_configured_remote(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(download_imagenet.shutil, "which", lambda name: "/usr/bin/rclone")
    monkeypatch.setattr(
        download_imagenet.subprocess,
        "run",
        lambda args, check: calls.append((args, check)),
    )

    download_imagenet.download_gdrive(tmp_path, "ann-drive:ANN-Project/imagenet")

    args, check = calls[0]
    assert args[:3] == [
        "rclone",
        "copy",
        "ann-drive:ANN-Project/imagenet/raw",
    ]
    assert args[3] == str(tmp_path)
    assert check is True


def test_credentials_are_required(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with pytest.raises(SystemExit, match="KAGGLE_USERNAME"):
        download_imagenet.download_kaggle(tmp_path)

    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        download_imagenet.download_hf(tmp_path)
