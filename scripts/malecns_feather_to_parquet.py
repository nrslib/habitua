#!/usr/bin/env python3
"""Convert the supplied MaleCNS Feather tables to Rust-readable Parquet tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as parquet


TABLES = {
    "annotations": (
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        (
            ("assignedOlHex1", "assigned_ol_hex1"),
            ("assignedOlHex2", "assigned_ol_hex2"),
            ("bodyId", "body_id"),
            ("type", "type_name"),
            ("class", "cell_class"),
            ("superclass", "superclass"),
            ("subclass", "subclass"),
            ("somaSide", "soma_side"),
            ("rootSide", "root_side"),
            ("status", "status"),
        ),
    ),
    "neurotransmitters": (
        "body-neurotransmitters-male-cns-v1.0.feather",
        (
            ("body", "body"),
            ("consensus_nt", "consensus_nt"),
            ("celltype_predicted_nt", "celltype_predicted_nt"),
        ),
    ),
    "connectome": (
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        (
            ("body_pre", "body_pre"),
            ("body_post", "body_post"),
            ("weight", "weight"),
        ),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def convert_table(input_path: Path, output_path: Path, columns: tuple[tuple[str, str], ...]) -> int:
    reader = ipc.open_file(input_path)
    source_indices = [reader.schema.get_field_index(source) for source, _ in columns]
    if any(index < 0 for index in source_indices):
        missing = [
            source
            for (source, _), index in zip(columns, source_indices)
            if index < 0
        ]
        raise ValueError(f"missing Feather columns in {input_path}: {missing}")

    writer: parquet.ParquetWriter | None = None
    row_count = 0
    try:
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index).select(source_indices)
            batch = batch.rename_columns([target for _, target in columns])
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = parquet.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                    use_dictionary=[
                        name
                        for name in table.column_names
                        if name not in {"body_id", "body", "body_pre", "body_post", "weight"}
                    ],
                )
            writer.write_table(table)
            row_count += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    for table_name, (file_name, columns) in TABLES.items():
        input_path = input_dir / file_name
        output_path = output_dir / f"{table_name}.parquet"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        row_count = convert_table(input_path, output_path, columns)
        sources.append(
            {
                "name": file_name,
                "path": str(input_path),
                "size_bytes": input_path.stat().st_size,
                "sha256": sha256(input_path),
            }
        )
        outputs.append(
            {
                "name": output_path.name,
                "path": str(output_path),
                "rows": row_count,
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
                "columns": [target for _, target in columns],
            }
        )

    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "MaleCNS v1.0",
                "source_directory": str(input_dir),
                "sources": sources,
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for output in outputs:
        print(
            f"{output['name']}: rows={output['rows']} "
            f"bytes={output['size_bytes']} sha256={output['sha256']}"
        )


if __name__ == "__main__":
    main()
