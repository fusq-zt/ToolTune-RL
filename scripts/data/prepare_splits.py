"""Freeze question/document-disjoint pools before constructing SFT trajectories.

Source IDs use official IDs when supplied and source revision + content hash
otherwise. No model score influences inclusion. Synthetic worlds stay grouped.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def boxed(s):
    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", s)]
    for start in reversed(starts):
        depth = 1
        for i in range(start, len(s)):
            depth += (s[i] == "{") - (s[i] == "}")
            if depth == 0:
                return s[start:i]
    return ""


def shingles(s):
    s = normalized(s)
    return {s[i : i + 5] for i in range(max(len(s) - 4, 1))}


def make_task(repo, revision, row, path):
    documents = []
    if repo == "openai/gsm8k":
        question, answer, kind = (
            row["question"],
            row["answer"].rsplit("####", 1)[-1].strip(),
            "numeric",
        )
    elif repo == "EleutherAI/hendrycks_math":
        question, answer, kind = row["problem"], boxed(row["solution"]), "math"
    else:
        question, answer, kind = row["question"], row["answer"], "qa"
        context = row["context"]
        for title, sentences in zip(context["title"], context["sentences"]):
            body = " ".join(sentences)
            documents.append(
                {"document_id": digest(normalized(title)), "title": title, "text": body}
            )
    content_hash = digest(normalized(question))
    source_id = str(row.get("id") or row.get("_id") or content_hash)
    return dict(
        task_id=repo + ":" + source_id,
        question=question.strip(),
        answer=answer,
        verifier=kind,
        source=repo,
        revision=revision,
        source_file=path,
        content_hash=content_hash,
        documents=documents,
        cluster_id=repo + ":" + source_id,
        required_retrieval=False,
    )


def synthetic(world, variant=0, distractors=2):
    # Arithmetic template families are partition-specific; paired worlds share ID.
    d = bytes.fromhex(digest(f"toolforge-v2:{world}:{variant}"))
    a, b, c = [int.from_bytes(d[i : i + 2], "big") % 800 + 10 for i in (0, 2, 4)]
    names = ["Entity-" + digest(f"{world}:{k}")[:10] for k in range(3)]
    operation = int(world.split("-")[-1]) % 6
    templates = [
        (
            f"What is the sum of the recorded values for {names[0]} and {names[1]}?",
            a + b,
        ),
        (f"Subtract the recorded value of {names[1]} from that of {names[0]}.", a - b),
        (f"Multiply the recorded values of {names[0]} and {names[1]}.", a * b),
        (
            f"Find twice the recorded value of {names[0]} plus the value of {names[1]}.",
            2 * a + b,
        ),
        (f'What is the largest recorded value among {", ".join(names)}?', max(a, b, c)),
        (
            f"Find the sum of the recorded values of {names[0]} and {names[1]}, minus {names[2]}.",
            a + b - c,
        ),
    ]
    split = world.rsplit("-", 1)[0]
    x, y, z = names
    reserved = {
        "rl": [
            (f"Add the recorded value of {x} to twice that of {y}.", a + 2 * b),
            (f"Subtract twice the recorded value of {y} from that of {x}.", a - 2 * b),
            (
                f"Multiply the recorded values of {x} and {y}, then add that of {z}.",
                a * b + c,
            ),
            (f"Add three times the recorded value of {x} to that of {y}.", 3 * a + b),
            (
                f"What is the smallest recorded value among {x}, {y}, and {z}?",
                min(a, b, c),
            ),
            (f"Sum the three recorded values of {x}, {y}, and {z}.", a + b + c),
        ],
        "dev_config": [
            (f"Double the sum of the recorded values of {x} and {y}.", 2 * a + 2 * b),
            (
                f"Subtract the value of {y} from twice the recorded value of {x}.",
                2 * a - b,
            ),
            (
                f"Multiply the recorded values of {x} and {y}, then subtract that of {z}.",
                a * b - c,
            ),
            (f"Add the recorded value of {x} to three times that of {y}.", a + 3 * b),
            (
                f"Find the range (largest minus smallest) of the recorded values of {x}, {y}, and {z}.",
                max(a, b, c) - min(a, b, c),
            ),
            (
                f"Double the recorded value of {x}, add that of {y}, then subtract that of {z}.",
                2 * a + b - c,
            ),
        ],
        "dev_checkpoint": [
            (
                f"Add three times the recorded value of {x} to twice that of {y}.",
                3 * a + 2 * b,
            ),
            (
                f"Subtract the recorded value of {y} from three times that of {x}.",
                3 * a - b,
            ),
            (
                f"Add the recorded values of {x} and {y}, then multiply by that of {z}.",
                (a + b) * c,
            ),
            (
                f"Add twice the recorded value of {x} to three times that of {y}.",
                2 * a + 3 * b,
            ),
            (
                f"Find the absolute difference of the recorded values of {x} and {y}.",
                abs(a - b),
            ),
            (
                f"Add the recorded value of {x} to twice that of {y}, then subtract that of {z}.",
                a + 2 * b - c,
            ),
        ],
        "test": [
            (f"Triple the sum of the recorded values of {x} and {y}.", 3 * a + 3 * b),
            (
                f"Subtract twice the recorded value of {y} from three times that of {x}.",
                3 * a - 2 * b,
            ),
            (
                f"Subtract the recorded value of {y} from that of {x}, then multiply by that of {z}.",
                (a - b) * c,
            ),
            (
                f"Add three times the recorded value of {x} to four times that of {y}.",
                3 * a + 4 * b,
            ),
            (
                f"Add the largest and smallest recorded values among {x}, {y}, and {z}.",
                max(a, b, c) + min(a, b, c),
            ),
            (
                f"Add the recorded values of {x} and {y}, then subtract twice that of {z}.",
                a + b - 2 * c,
            ),
        ],
    }
    question, answer = reserved.get(split, templates)[operation]
    docs = [
        {
            "document_id": f"{world}:{i}",
            "title": f"Record {n}",
            "text": f"The recorded value of {n} is {v}.",
        }
        for i, (n, v) in enumerate(zip(names, (a, b, c)))
    ]
    docs += [
        {
            "document_id": f"{world}:noise{i}",
            "title": f"Unrelated archival record {i}",
            "text": f"An unrelated marker has value {i*97+13}. This is not any named entity's value.",
        }
        for i in range(distractors)
    ]
    return dict(
        task_id=f"synthetic:{world}:v{variant}",
        question=question,
        answer=str(answer),
        verifier="numeric",
        source="toolforge/synthetic_v2",
        revision="v2.1",
        content_hash=digest(normalized(question)),
        documents=docs,
        family="retrieve_then_compute",
        cluster_id="world:" + world,
        required_retrieval=True,
        world_id=world,
        template_id=f"{split}:{operation}",
        generation_seed_id=f"{world}:{variant}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "data/prepared"
    if (out / "manifest.json").exists():
        raise RuntimeError("Manifest already frozen; do not overwrite it")
    out.mkdir(parents=True, exist_ok=True)
    pools = {"gsm": [], "math": [], "retrieval": []}
    for repo, pool in [
        ("openai/gsm8k", "gsm"),
        ("EleutherAI/hendrycks_math", "math"),
        ("hotpotqa/hotpot_qa", "retrieval"),
    ]:
        metadata = json.loads(
            (
                args.root / "sources/metadata" / (repo.replace("/", "--") + ".json")
            ).read_text()
        )
        base = args.root / "data/raw" / repo.replace("/", "--")
        for path in sorted(base.rglob("*train*.parquet")):
            for row in pq.read_table(path).to_pylist():
                task = make_task(
                    repo, metadata["sha"], row, str(path.relative_to(base))
                )
                if (
                    task["question"]
                    and task["answer"]
                    and (pool != "retrieval" or task["documents"])
                ):
                    pools[pool].append(task)
        pools[pool].sort(key=lambda x: digest("42:" + x["task_id"]))
    used, doc_splits, excluded = set(), {}, Counter()
    lsh = MinHashLSH(threshold=0.70, num_perm=64)
    texts = {}
    selected = []

    def take(pool, n, split, family):
        kept = []
        for row in pools[pool]:
            if len(kept) >= n:
                break
            h = row["content_hash"]
            if h in used:
                continue
            doc_ids = {d["document_id"] for d in row["documents"]}
            if any(d in doc_splits and doc_splits[d] != split for d in doc_ids):
                excluded["shared_document_cross_split"] += 1
                continue
            chunks = shingles(row["question"])
            mh = MinHash(num_perm=64, seed=42)
            mh.update_batch([x.encode() for x in sorted(chunks)])
            near = any(
                len(chunks & texts[k]) / max(len(chunks | texts[k]), 1) >= 0.85
                for k in lsh.query(mh)
            )
            if near:
                excluded["near_duplicate"] += 1
                continue
            lsh.insert(h, mh)
            texts[h] = chunks
            used.add(h)
            for doc in doc_ids:
                doc_splits[doc] = split
            row = dict(row, split=split, family=family)
            kept.append(row)
        if len(kept) != n:
            raise RuntimeError(
                f"Insufficient document-disjoint {split}/{family}: {len(kept)}/{n}"
            )
        return kept

    # Reserve evaluation first using only source data. Never evaluate these during construction.
    for split, size in [
        ("test", 2000),
        ("dev_config", 400),
        ("dev_checkpoint", 400),
        ("rl", 4000),
        ("sft_pool", 8000),
    ]:
        rows = take("gsm", size // 5, split, "direct_answer")
        rows += take("gsm", size * 15 // 100, split, "python_math")
        rows += take("math", size * 15 // 100, split, "python_math")
        rows += take("retrieval", size * 3 // 10, split, "retrieval")
        rows += [dict(synthetic(f"{split}-{i}"), split=split) for i in range(size // 5)]
        rows.sort(key=lambda r: digest("2026:" + r["task_id"]))
        path = out / (split + ".jsonl")
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        selected += rows
        print(split, len(rows), flush=True)
    # Cluster IDs for shared documents within each partition (including distractors).
    parent = {r["task_id"]: r["task_id"] for r in selected}
    owner = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in selected:
        for doc in row["documents"]:
            key = doc["document_id"]
            if key in owner:
                a, b = find(row["task_id"]), find(owner[key])
                parent[max(a, b)] = min(a, b)
            else:
                owner[key] = row["task_id"]
    for row in selected:
        row["cluster_id"] = find(row["task_id"])
    files = {}
    for split in {r["split"] for r in selected}:
        path = out / (split + ".jsonl")
        path.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for r in selected
                if r["split"] == split
            )
        )
        files[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": sum(r["split"] == split for r in selected),
        }
    manifest = {
        "status": "frozen_pre_model",
        "files": files,
        "excluded_counts": dict(excluded),
        "grouping": "all provided document titles + question exact/approximate near-duplicates + synthetic worlds",
        "near_duplicate_audit": {
            "method": "MinHash64 LSH candidate retrieval; exact character5gram Jaccard>=0.85",
            "recall_not_guaranteed": True,
        },
        "public_pretraining_contamination": "not ruled out",
        "test_synthetic_fraction": 0.20,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
