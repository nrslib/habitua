#!/usr/bin/env python3
"""Prepare the frozen helper artifact and a separately distributed rate pack."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


PACK_FILES = (
    "rate_manifest.json",
    "rate-neurons.json",
    "rate-out.bin",
    "rate-in.bin",
    "rate-populations.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_artifact(source: Path, output_dir: Path) -> None:
    original = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(original, dict) or not Path(original["pack_path"]).is_absolute():
        raise ValueError("source artifact must have an absolute pack_path")
    sanitized = dict(original)
    sanitized["pack_path"] = "rate-full"
    serialized = json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n"
    target = output_dir / source.name
    output_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(serialized, encoding="utf-8")
    generated = json.loads(target.read_text(encoding="utf-8"))
    if generated.pop("pack_path") != "rate-full" or generated != {
        key: value for key, value in original.items() if key != "pack_path"
    }:
        raise ValueError("artifact learning content changed")
    for value in all_strings(generated):
        if value.startswith("/") or "/Users/" in value:
            raise ValueError("artifact contains an absolute path outside pack_path")
    digest = sha256(target)
    (output_dir / (target.name + ".sha256")).write_text(
        f"{digest}  {target.name}\n", encoding="utf-8"
    )
    print(f"artifact SHA-256: {digest}; all fields except pack_path match source")


def all_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_strings(child)
    elif isinstance(value, str):
        yield value


def prepare_pack(pack_dir: Path, artifact_path: Path, output_dir: Path) -> None:
    if not pack_dir.is_dir() or any(not (pack_dir / name).is_file() for name in PACK_FILES):
        raise ValueError("rate pack is incomplete")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest_path = pack_dir / "rate_manifest.json"
    pack_manifest_sha256 = sha256(manifest_path)
    if artifact["pack_manifest_sha256"] != pack_manifest_sha256:
        raise ValueError("rate pack manifest does not match the distribution artifact")
    pack_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "malecns-v1.0-rate-full.tar"
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name in PACK_FILES:
            path = pack_dir / name
            info = tar.gettarinfo(str(path), arcname=f"rate-full/{name}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as stream:
                tar.addfile(info, stream)
    manifest = {
        "schema": "habitua-rate-pack-distribution-v1",
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "pack_manifest_sha256": pack_manifest_sha256,
        "source": "MaleCNS v1.0, Janelia Research Campus",
        "source_url": "https://male-cns.janelia.org/download/",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "transformation": "Converted MaleCNS v1.0 annotations and connectivity into habitua rate-full neuron metadata, outgoing/incoming edge arrays, populations, and a verification manifest; archived those files without further transformation.",
        "normalization": pack_manifest["normalization"],
        "sign_rule": pack_manifest["sign_rule"],
    }
    write_json(output_dir / "malecns-v1.0-rate-full.distribution.json", manifest)
    print(f"pack archive SHA-256: {manifest['archive_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    artifact = commands.add_parser("artifact")
    artifact.add_argument("source", type=Path)
    artifact.add_argument("output_dir", type=Path)
    pack = commands.add_parser("pack")
    pack.add_argument("pack_dir", type=Path)
    pack.add_argument("artifact", type=Path)
    pack.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.command == "artifact":
        prepare_artifact(args.source, args.output_dir)
    else:
        prepare_pack(args.pack_dir, args.artifact, args.output_dir)


if __name__ == "__main__":
    main()
