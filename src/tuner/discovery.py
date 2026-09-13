from __future__ import annotations

import argparse
import ast
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CURATED = {
    "ServiceClient.get_server_capabilities": "capabilities_get",
    "RestClient.list_training_runs": "training_list",
    "RestClient.get_training_run": "training_get",
    "RestClient.list_checkpoints": "checkpoint_list",
    "RestClient.get_weights_info_by_tinker_path": "checkpoint_get",
    "SamplingClient.sample": "sample",
    "SamplingClient.compute_logprobs": "compute_logprobs",
    "supervised.train.main": "train_sft",
    "eval.benchmarks.run_benchmark": "evaluate",
}

ADVANCED_METHODS = {
    "forward",
    "forward_backward",
    "forward_backward_custom",
    "optim_step",
    "save_state",
    "load_state",
    "load_state_with_optimizer",
    "save_weights_for_sampler",
}

UNSAFE_METHODS = {
    "delete_checkpoint",
    "delete_checkpoint_from_tinker_path",
    "publish_checkpoint_from_tinker_path",
    "unpublish_checkpoint_from_tinker_path",
    "assign_session_project",
}


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    return {
        "parameters": ast.unparse(node.args),
        "returns": ast.unparse(node.returns) if node.returns else None,
        "async": isinstance(node, ast.AsyncFunctionDef),
        "decorators": [ast.unparse(item) for item in node.decorator_list],
    }


def _public_symbols(root: Path, source: str) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        if path.name.endswith("_test.py") or "tests" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join([root.name, *parts])
        declared_exports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                try:
                    value = ast.literal_eval(node.value)
                    if isinstance(value, (list, tuple)):
                        declared_exports.update(item for item in value if isinstance(item, str))
                except (ValueError, TypeError):
                    pass
        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("_"):
                entry = _entry(source, f"{module}.{node.name}", node.name, path, root)
                symbols.append({**entry, "kind": "function", **_signature(node)})
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                entry = _entry(source, f"{module}.{node.name}", node.name, path, root)
                fields = {
                    child.target.id: {
                        "annotation": ast.unparse(child.annotation),
                        "default": ast.unparse(child.value) if child.value else None,
                    }
                    for child in node.body
                    if isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and not child.target.id.startswith("_")
                }
                symbols.append(
                    {
                        **entry,
                        "kind": "class",
                        "fields": fields,
                        "bases": [ast.unparse(item) for item in node.bases],
                        "decorators": [ast.unparse(item) for item in node.decorator_list],
                    }
                )
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                        not child.name.startswith("_") or child.name == "__init__"
                    ):
                        entry = _entry(
                            source, f"{module}.{node.name}.{child.name}", child.name, path, root
                        )
                        symbols.append({**entry, "kind": "method", **_signature(child)})
            elif isinstance(node, ast.ImportFrom):
                package = module if path.name == "__init__.py" else module.rpartition(".")[0]
                if node.level:
                    prefix = package.split(".")
                    prefix = prefix[: len(prefix) - node.level + 1]
                    target_module = ".".join([*prefix, *([node.module] if node.module else [])])
                else:
                    target_module = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name in declared_exports or alias.asname == alias.name:
                        entry = _entry(source, f"{module}.{name}", name, path, root)
                        symbols.append(
                            {**entry, "kind": "export", "target": f"{target_module}.{alias.name}"}
                        )
        for entry in symbols:
            if entry["file"] == path.relative_to(root.parent).as_posix():
                entry["explicit_export"] = (
                    entry["symbol"].removeprefix(module + ".") in declared_exports
                )
    identities = {item["symbol"] for item in symbols}
    for entry in symbols:
        symbol = entry["symbol"]
        paired = symbol[:-6] if symbol.endswith("_async") else symbol + "_async"
        if paired in identities:
            entry["paired_symbol"] = paired
    return symbols


def _entry(source: str, symbol: str, method: str, path: Path, root: Path) -> dict[str, Any]:
    matched = next(((key, tool) for key, tool in CURATED.items() if symbol.endswith(key)), None)
    if matched:
        classification, tool = "curated", matched[1]
    elif method in UNSAFE_METHODS:
        classification, tool = "unsafe", None
    elif method in ADVANCED_METHODS:
        classification, tool = "advanced", None
    else:
        classification, tool = "unclassified", None
    return {
        "source": source,
        "symbol": symbol,
        "file": path.relative_to(root.parent).as_posix(),
        "classification": classification,
        "mcp_tool": tool,
    }


def generate(workspace: Path) -> dict[str, Any]:
    tinker_repo = workspace / "tinker"
    cookbook_repo = workspace / "tinker-cookbook"
    symbols = _public_symbols(tinker_repo / "src" / "tinker", "tinker")
    symbols += _public_symbols(cookbook_repo / "tinker_cookbook", "tinker-cookbook")
    symbols.sort(key=lambda item: (item["source"], item["symbol"]))
    counts: dict[str, int] = {}
    for item in symbols:
        counts[item["classification"]] = counts.get(item["classification"], 0) + 1
    return {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "upstream": {
            "tinker": {"commit": _git_commit(tinker_repo)},
            "tinker-cookbook": {"commit": _git_commit(cookbook_repo)},
        },
        "discovery": {
            "direct_ast": "executed",
            "code2mcp": "reviewed_as_secondary_oracle_not_imported",
            "mcpify": "reviewed_as_secondary_oracle_not_imported",
            "note": "Generated tools never define Tuner product contracts.",
        },
        "counts": counts,
        "symbols": symbols,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh or verify the Tuner upstream API manifest"
    )
    parser.add_argument(
        "--check", action="store_true", help="Fail if upstream symbols or commits drifted"
    )
    args = parser.parse_args()
    tuner_root = Path(__file__).resolve().parents[2]
    workspace = tuner_root.parent
    output = tuner_root / "generated" / "tinker_api_manifest.json"
    manifest = generate(workspace)
    if args.check:
        if not output.is_file():
            raise SystemExit("Manifest is missing; run tuner-discover")
        existing = json.loads(output.read_text(encoding="utf-8"))
        for key in ("schema_version", "upstream", "counts", "symbols"):
            if existing.get(key) != manifest.get(key):
                raise SystemExit(
                    f"API manifest drift detected in {key}; run tuner-discover and classify changes"
                )
        print("API manifest is current")
        return
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
