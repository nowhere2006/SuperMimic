"""Download the pinned 40-clip G1 LAFAN1 release and prepare local mjlab NPZs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
MJLAB_COMMIT = "8bb2bc9e187797689ec82c78c35019afd2fd0376"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2) + "\n")
    part.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/lafan1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/lafan1")
    parser.add_argument("--direct", action="store_true", help="bypass proxy for this command only")
    parser.add_argument("--local-only", action="store_true",help="rebuild solely from verified local source-index; no network")
    args = parser.parse_args()
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    from supermimic_study.data.motion_conversion import G1MotionConverter, PREPROCESS_VERSION, sha256
    source = metadata.distribution("mjlab").read_text("direct_url.json")
    if not source or MJLAB_COMMIT not in source:
        raise RuntimeError("preparation requires the pinned mjlab commit")

    def read_url(url: str, maximum: int) -> bytes:
        for attempt in range(3):
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.direct else urllib.request.build_opener()
                with opener.open(url, timeout=25) as stream:
                    content = stream.read(maximum + 1)
                if len(content) > maximum:
                    raise ValueError("download exceeds expected size bound")
                return content
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1)
        raise RuntimeError("unreachable download state")

    if args.local_only:
        index = json.loads((args.raw_dir/"source-index.json").read_text())
        if index.get("dataset") != DATASET or index.get("revision") != REVISION:
            raise ValueError("local source index has unexpected dataset/revision")
        tree = [{"type":"file","path":x["file"],"size":x["bytes"],"oid":x["git_blob_oid"]}
                for x in index["files"]]
    else:
        tree_url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/g1"
        tree = json.loads(read_url(tree_url, 1024 * 1024))
    files = sorted((x for x in tree if x["type"] == "file" and x["path"].endswith(".csv")), key=lambda x: x["path"])
    if len(files) != 40 or sum(x["size"] for x in files) != 90_487_276:
        raise ValueError("pinned release inventory differs from the verified 40-clip G1 release")

    def download(item: dict) -> dict:
        relative = Path(item["path"])
        if relative.parent != Path("g1") or relative.name != item["path"].split("/")[-1]:
            raise ValueError("unexpected source path")
        destination = args.raw_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{item['path']}"
        if args.local_only and not destination.is_file():
            raise ValueError(f"missing local source: {destination}")
        content = destination.read_bytes() if destination.exists() else read_url(url, item["size"])
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if len(content) != item["size"] or blob != item["oid"]:
            raise ValueError(f"source size/blob verification failed: {destination}")
        if not destination.exists():
            part = destination.with_suffix(".csv.part")
            part.write_bytes(content)
            part.rename(destination)
        info = {"dataset": DATASET, "revision": REVISION, "file": item["path"], "url": url,
                "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "git_blob_oid": blob}
        write_json(destination.with_suffix(".source.json"), info)
        return info

    sources = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, future in enumerate(as_completed([pool.submit(download, x) for x in files]), 1):
            info = future.result()
            sources.append(info)
            print(f"Source {n}/40 verified: {info['file']}", flush=True)
    sources.sort(key=lambda x: x["file"])
    write_json(args.raw_dir / "source-index.json", {"dataset": DATASET, "revision": REVISION, "files": sources})
    converter = G1MotionConverter.from_mjlab()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, info in enumerate(sources, 1):
        raw = args.raw_dir / info["file"]
        destination = args.output_dir / (raw.stem + ".npz")
        sidecar = destination.with_suffix(".conversion.json")
        if destination.exists():
            if not sidecar.is_file():
                raise ValueError(f"existing output has no conversion record: {destination}")
            row = json.loads(sidecar.read_text())
            if (row.get("source_sha256") != info["sha256"] or row.get("preprocess_version") != PREPROCESS_VERSION
                    or row.get("npz_sha256") != sha256(destination)):
                raise ValueError(f"existing output provenance mismatch: {destination}")
            print(f"Converted {n}/40 verified existing: {raw.stem}", flush=True)
        else:
            started = time.perf_counter()
            row = converter.convert(raw, destination)
            row.update(preprocess_version=PREPROCESS_VERSION, conversion_seconds=time.perf_counter() - started)
            write_json(sidecar, row)
            print(f"Converted {n}/40: {raw.stem}, {row['frames']} frames, {row['conversion_seconds']:.1f}s", flush=True)
        rows.append(row)
    manifest = {
        "schema_version": 1, "dataset": DATASET, "revision": REVISION,
        "input_fps": 30, "fps": 50, "preprocess_version": PREPROCESS_VERSION,
        "resampling": "float64 time grid; normalized shortest-arc quaternion SLERP/nlerp; centered world-frame SO(3) angular velocity",
        "joint_names": list(converter.joint_names), "body_names": list(converter.body_names),
        "mjlab_source": json.loads(source), "conversion_backend": "native MuJoCo CPU position/velocity kinematics; no physics rollout",
        "versions": {x: metadata.version(x) for x in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "numpy")},
        "clips": rows, "total_frames": sum(r["frames"] for r in rows),
        "duration_s": sum(r["duration_s"] for r in rows), "total_bytes": sum(r["bytes"] for r in rows),
        "split": "all 40 available G1 reference motions; no training/evaluation split asserted",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"READY: {len(rows)} clips, {manifest['total_frames']} frames, {manifest['duration_s']/3600:.6f} h", flush=True)


if __name__ == "__main__":
    main()
