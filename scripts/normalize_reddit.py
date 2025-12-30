import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def dedupe(records: Iterable[Dict]) -> Iterable[Dict]:
    seen: Set[Tuple] = set()
    for rec in records:
        key = (
            rec.get("platform"),
            rec.get("post_id"),
            rec.get("url"),
            rec.get("text"),
        )
        if key in seen:
            continue
        seen.add(key)
        yield rec


def save_jsonl(records: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize(inputs, output: Path) -> int:
    all_records = []
    for pattern in inputs:
        for file in glob.glob(pattern):
            for rec in read_jsonl(Path(file)):
                all_records.append(rec)
    deduped = list(dedupe(all_records))
    save_jsonl(deduped, output)
    return len(deduped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fusion et déduplication des JSONL Reddit.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["Projet_API/data/raw/*.jsonl"],
        help="Fichiers ou motifs glob de JSONL bruts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Projet_API/data/clean/reddit_clean.jsonl"),
        help="Fichier JSONL de sortie dédupliqué.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = normalize(args.inputs, args.output)
    print(f"Fusionné et dédupliqué: {count} posts -> {args.output}")
