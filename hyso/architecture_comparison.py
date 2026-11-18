"""Multi-architecture metadata comparison; separate from binary evidence comparison."""

from collections import Counter

from .model_checks import config_digest, count_parameters, preflight_model
from .models import ModelRegistry


def compare_architectures(configs, *, names=None, exact_parameters=False):
    """Compare 2..32 schema1/2/3 descriptions without changing them.

    Default: allocation-free Core metadata only. ``exact_parameters=True`` opts
    into the <=1M-bound CPU numel path, never inference or training. Invalid
    descriptions retain diagnostics and no runtime counts. No performance or
    quality value is derived from architecture metadata.
    """
    if not isinstance(configs, (list, tuple)) or not 2 <= len(configs) <= 32:
        raise ValueError("Provide 2..32 architecture configurations")
    if type(exact_parameters) is not bool:
        raise TypeError("exact_parameters must be bool")
    if names is None:
        names = [f"model_{i + 1}" for i in range(len(configs))]
    if len(names) != len(configs) or any(type(n) is not str or not n for n in names):
        raise ValueError("Provide one nonempty name per architecture")
    categories = {s.name: s.category for s in ModelRegistry().list()}
    categories.update(
        {k: "graph_fusion" for k in ("add", "concat", "weighted_sum", "multiply")}
    )
    columns = []
    for name, config in zip(names, configs):
        report = preflight_model(config)
        if not report["valid"]:
            # Core owns validation. Do not inspect rejected head/node containers
            # or present partial geometry/counts as a valid architecture.
            columns.append(
                dict(
                    name=name,
                    config_sha256=config_digest(config),
                    valid=False,
                    inputs=None,
                    outputs=None,
                    output_shape=None,
                    node_count=None,
                    connections=None,
                    branch_count=None,
                    mechanisms={},
                    categories={},
                    parameter_count=None,
                    parameter_bound=None,
                    parameter_status="UNAVAILABLE",
                    parameter_reason="Invalid architecture; inspect Core diagnostics",
                    activation_estimate=None,
                    diagnostics=report["diagnostics"],
                )
            )
            continue
        graph = config.get("schema_version") == 3
        expanded = graph or config.get("kind") in (
            "sequential",
            "sequential_classifier",
        )
        nodes = config.get("nodes", []) if graph else config.get("blocks", [])
        mechanisms = Counter(
            n.get("component", n.get("name")) for n in nodes if isinstance(n, dict)
        )
        if not graph and config.get("head"):
            mechanisms[config["head"]["name"]] += 1
        if config.get("kind") in ("registered_model", "modular_cnn"):
            mechanisms[config.get("name", "modular_cnn")] += 1
        connections = report.get("edges", [])
        consumers = Counter(e["source"] for e in connections)
        exact = report.get("parameter_count") if report["valid"] else None
        reason = (
            "Core static count"
            if exact is not None
            else "No exact count requested; inspect Core conservative bound"
        )
        if exact_parameters and report["valid"]:
            try:
                exact = count_parameters(config)
                reason = "CPU nn.Module numel; no forward pass"
            except ValueError as error:
                reason = str(error)
        if not report["valid"]:
            reason = "Invalid architecture; inspect Core diagnostics"
        columns.append(
            dict(
                name=name,
                config_sha256=config_digest(config),
                valid=report["valid"],
                inputs=config.get("inputs", {"input": report.get("input_shape")}),
                outputs=report.get("outputs", {}),
                output_shape=report.get("output_shape"),
                node_count=len(nodes) if expanded else None,
                connections=connections if expanded else None,
                branch_count=sum(v > 1 for v in consumers.values())
                if graph
                else 0
                if expanded
                else None,
                mechanisms=dict(mechanisms),
                categories=dict(
                    Counter(
                        {
                            k: sum(
                                v
                                for n, v in mechanisms.items()
                                if categories.get(n, "model") == k
                            )
                            for k in {categories.get(n, "model") for n in mechanisms}
                        }
                    )
                ),
                parameter_count=exact,
                parameter_bound=report.get("parameter_estimate"),
                parameter_status="EXACT" if exact is not None else "UNAVAILABLE",
                parameter_reason=reason,
                activation_estimate=report.get("activation_estimate"),
                diagnostics=report["diagnostics"],
            )
        )
    for col in columns:
        col["parameter_delta"] = (
            col["parameter_count"] - columns[0]["parameter_count"]
            if col["parameter_count"] is not None
            and columns[0]["parameter_count"] is not None
            else None
        )
    groups = {}
    for i, col in enumerate(columns):
        groups.setdefault(col["config_sha256"], []).append(i)
    return dict(
        format="hyso.architecture_comparison",
        schema_version=1,
        columns=columns,
        equivalent_groups=[g for g in groups.values() if len(g) > 1],
        input_geometry_matches=all(
            c["inputs"] is not None and c["inputs"] == columns[0]["inputs"]
            for c in columns
        ),
        limitations=[
            "Static metadata and optional CPU parameter counts only; no quality/speed ranking",
            "Configuration identity does not identify trained weights",
        ],
    )
