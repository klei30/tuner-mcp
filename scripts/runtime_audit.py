"""No-credit import and native-config audit for the built Tuner runtime."""

from __future__ import annotations

import importlib
import json

from tuner.evaluation import benchmark_catalog
from tuner.recipes import CATALOG, construct_recipe_config, recipe_list


def main() -> None:
    recipes = {item["recipe"]: item for item in recipe_list()}
    recipe_results = []
    operational = {"log_path", "log_root", "experiment_dir"}
    for descriptor in CATALOG:
        info = recipes[descriptor.recipe]
        required = [
            name
            for name, field in info["config_schema"].get("properties", {}).items()
            if field.get("required") and name not in operational
        ]
        result = {
            "recipe": descriptor.recipe,
            "status": info["status"],
            "requirements": info["requirements"],
            "required_config": required,
        }
        if not info["requirements"] and not required:
            try:
                construct_recipe_config(descriptor, {}, "/tmp/tuner-runtime-audit")
                result["default_config"] = "valid"
            except Exception as exc:
                result["default_config"] = f"error:{type(exc).__name__}:{exc}"
        else:
            result["default_config"] = "requires_input"
        recipe_results.append(result)

    benchmark_results = []
    for benchmark in benchmark_catalog()["benchmarks"]:
        module = f"tinker_cookbook.eval.benchmarks.{benchmark['module']}"
        try:
            importlib.import_module(module)
            status = "imported"
        except Exception as exc:
            status = f"error:{type(exc).__name__}:{exc}"
        benchmark_results.append({"name": benchmark["name"], "module": module, "status": status})

    print(
        json.dumps(
            {
                "recipes": recipe_results,
                "benchmarks": benchmark_results,
                "recipe_count": len(recipe_results),
                "benchmark_count": len(benchmark_results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
