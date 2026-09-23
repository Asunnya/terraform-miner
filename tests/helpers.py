import json
import os


def make_record(**overrides):
    """Registro no formato de data.jsonl (um por commit+arquivo)."""
    record = {
        "repo": "owner_repo",
        "commit_hash": "aaa111",
        "author": "dev",
        "date": "2024-01-15T10:00:00+00:00",
        "message": "fix: something",
        "file": "main.tf",
        "patch": "@@ -1 +1 @@\n-a\n+b\n",
        "head_hash": "fff999",
    }
    record.update(overrides)
    return record


def write_jsonl(path, records, extra_lines=()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        for line in extra_lines:
            f.write(line + "\n")
    return path
