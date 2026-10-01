#!/usr/bin/env python3
"""Extract a bounded soma sample joined to an existing rate graph by body ID."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.feather as feather

parser = argparse.ArgumentParser()
parser.add_argument("--annotations", type=Path, required=True)
parser.add_argument("--artifact", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
artifact = json.loads(args.artifact.read_text())
neurons = json.loads((Path(artifact["pack_path"]) / "rate-neurons.json").read_text())
rows = feather.read_table(args.annotations, columns=["bodyId", "somaLocation"]).to_pylist()
positions = {r["bodyId"]: r["somaLocation"] for r in rows if r["somaLocation"] is not None and len(r["somaLocation"]) == 3}
positioned = [(str(n["body_id"]), positions[n["body_id"]]) for n in neurons if n["body_id"] in positions]
count = min(1024, len(positioned))
sample = [positioned[i * len(positioned) // count] for i in range(count)]
result = {"schema": "malecns-soma-sample-v1", "graph_fingerprint": artifact["graph_fingerprint"], "total_neurons": len(neurons), "positioned_neurons": len(positioned), "source_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(), "neurons": [{"body_id": body, "position": position} for body, position in sample]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
print(f"{count} / {len(positioned)} positioned / {len(neurons)} neurons; {args.output.stat().st_size} bytes")
