"""Read-only observations of native Python results and Core model diagnostics.

Threshold rules migrated from the retired UI. Findings are hypotheses, never
causal proof or automatic model changes. No storage, service or workload launch.
Use existing analysis.compare_models for authoritative binary evidence comparison.
"""

from dataclasses import dataclass
import hashlib
import json
import math

from .model_checks import config_digest


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _bilingual(tr, en):
    return dict(tr=tr, en=en)


@dataclass(frozen=True)
class ResultObservation:
    """Validated native result with explicit caller-declared architecture context."""

    _json: str


def observe_result(
    result,
    *,
    config,
    model_id,
    dataset=None,
    split=None,
    observed_at=None,
    head_mapping=None,
):
    """Bind native Training/Evaluation/Performance results without touching a store.

    Identity labels/data split are caller declarations, not producer proof.
    Native constructors validate payload consistency. Chronological comparison
    requires explicit UTC observation time; missing context prevents comparisons.
    """
    from .experiments.workflow import _restore_result
    from .analysis._native import restore_native

    if not hasattr(result, "to_dict"):
        raise TypeError("A native result object with to_dict() is required")
    payload = result.to_dict()
    if payload.get("format") == "hyso.training_session_result":
        _restore_result("training_result", payload)
        kind = "training"
    else:
        restore_native(payload)
        kind = (
            "evaluation"
            if payload.get("format") == "hyso.evaluation.classification"
            else "performance"
        )
    if type(model_id) is not str or not model_id:
        raise ValueError("An explicit model_id is required")
    if observed_at is not None:
        from datetime import datetime

        parsed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("observed_at requires an explicit UTC time")
    provenance = dict(
        config_sha256=config_digest(config),
        model_id=model_id,
        run_id=model_id,
        dataset=dataset,
        split=split,
        observed_at=observed_at,
        head_mapping=head_mapping,
        device=payload.get("prepared_effective", {}).get(
            "device", payload.get("environment", {}).get("device")
        ),
    )
    embedded = payload.get("model", {}).get("model_config")
    if embedded is not None and config_digest(embedded) != provenance["config_sha256"]:
        raise ValueError("Native architecture differs from declared config")
    record = dict(
        id=_digest(dict(payload=payload, context=provenance)),
        payload=payload,
        kind=kind,
        payload_sha256=_digest(payload),
        provenance=provenance,
        authenticity="UNVERIFIED_SOURCE",
    )
    return ResultObservation(json.dumps(record, sort_keys=True, allow_nan=False))


def diagnose_results(observations):
    """Read native observations; no data capture, new execution or automatic repair."""
    if not isinstance(observations, (list, tuple)) or any(
        type(o) is not ResultObservation for o in observations
    ):
        raise TypeError("Provide ResultObservation objects made by observe_result")
    return _diagnose([json.loads(o._json) for o in observations])


def observe_model_checks(checks, *, model_id, observed_at=None):
    """Bind explicit Python smoke results; no pytest/JUnit or database envelope."""
    if checks.get("format") != "hyso.model_checks" or checks.get("schema_version") != 1:
        raise ValueError("Expected run_model_checks result")
    if any(c.get("status") not in ("PASS", "FAIL", "SKIP") for c in checks["cases"]):
        raise ValueError("Invalid check status")
    record = dict(
        id=_digest(dict(checks=checks, model_id=model_id, observed_at=observed_at)),
        kind="testing",
        payload=checks,
        payload_sha256=_digest(checks),
        authenticity="UNVERIFIED_SOURCE",
        provenance=dict(
            config_sha256=checks["config_sha256"],
            run_id=model_id,
            model_id=model_id,
            observed_at=observed_at,
            device="cpu",
            dataset=checks["fixture"],
            split="synthetic",
        ),
    )
    return ResultObservation(json.dumps(record, sort_keys=True, allow_nan=False))


def suggest_model_corrections(config):
    """Explain Core diagnostics; validate channel candidates on copies, never apply."""
    import copy
    from .model_checks import preflight_model

    report = preflight_model(config)
    items = []
    for diagnostic in report["diagnostics"]:
        explanation = explain_diagnostic(diagnostic, config, report)
        candidate = copy.deepcopy(config)
        node = next(
            (
                n
                for n in candidate.get("nodes", [])
                if n["id"] == diagnostic.get("layer_id")
            ),
            None,
        )
        verified = None
        if (
            node
            and diagnostic["code"] == "CHANNEL_MISMATCH"
            and "in_channels" in node["params"]
            and type(diagnostic.get("actual")) is int
        ):
            node["params"]["in_channels"] = diagnostic["actual"]
            verified = preflight_model(candidate)
        items.append(
            dict(
                diagnostic=diagnostic,
                explanation=explanation,
                candidate_config=candidate if verified and verified["valid"] else None,
                candidate_status="STATICALLY_VALIDATED"
                if verified and verified["valid"]
                else "NOT_VALIDATED",
                limitations="No runtime or quality effect measured; caller must review",
            )
        )
    return items


def explain_diagnostic(diagnostic, config, report):
    """Translate only conditions explicitly reported by Core; retain original evidence."""
    node_id = diagnostic.get("layer_id")
    node = next((row for row in config.get("nodes", []) if row["id"] == node_id), {})
    name = node.get("component")
    message = diagnostic.get("message_en", "")
    observed = _bilingual(diagnostic.get("message_tr", message), message)
    change = _bilingual(
        "Core tanısındaki beklenen ve alınan değerleri bu katmanın ayarlarıyla karşılaştırın.",
        "Compare the Core diagnostic's expected and actual values with this layer's settings.",
    )
    title = _bilingual("Katman sözleşmesi ihlali", "Layer contract violation")
    if (
        name == "reshape"
        and "reshape must preserve per-sample element count" in message
    ):
        title = _bilingual(
            "Reshape eleman sayısını korumuyor", "Reshape changes element count"
        )
        observed = _bilingual(
            f"İstenen yeni boyutta {diagnostic['actual']} eleman var; önceki katman örnek başına {diagnostic['expected']} eleman üretiyor.",
            f"The requested shape has {diagnostic['actual']} elements; the previous layer produces {diagnostic['expected']} elements per sample.",
        )
        change = _bilingual(
            "CHW boyutlarının çarpımını beklenen eleman sayısına eşitleyin. Batch ekseni değişmez.",
            "Set the product of CHW dimensions to the expected element count. The batch axis is preserved.",
        )
    elif (
        name == "pixel_unshuffle"
        and "H and W must be divisible by downscale_factor" in message
    ):
        factor = node.get("params", {}).get("downscale_factor")
        title = _bilingual(
            "PixelUnshuffle bölünebilirlik hatası", "PixelUnshuffle divisibility error"
        )
        observed = _bilingual(
            f"Genişlik ve yükseklik {diagnostic['actual']}, küçültme faktörü {factor} değerine tam bölünemiyor.",
            f"Spatial dimensions {diagnostic['actual']} are not both divisible by downscale factor {factor}.",
        )
        change = _bilingual(
            "Her iki uzamsal boyutu bölen bir faktör seçin veya önce açık bir Resize düğümü ekleyin.",
            "Choose a factor dividing both spatial dimensions, or add an explicit Resize node first.",
        )
    elif "head pooling requires NCHW, vector features require pooling=none" in message:
        title = _bilingual(
            "Başlığın havuzlama girdisi uyumsuz", "Head pooling input mismatch"
        )
        observed = _bilingual(
            "Başlığa vektör özellikler geliyor; seçili havuzlama NCHW görüntü özellikleri gerektiriyor.",
            "The head receives vector features; the selected pooling requires NCHW image features.",
        )
        change = _bilingual(
            "Düzleştirilmiş vektör için pooling=none seçin; in_features değerini Core beklenen değeriyle eşleştirin.",
            "Select pooling=none for flattened vectors and match in_features to the Core expected value.",
        )
    elif "prediction head in_features mismatch" in message:
        title = _bilingual(
            "Başlık özellik sayısı uyumsuz", "Head feature count mismatch"
        )
        observed = _bilingual(
            f"Başlık {diagnostic['actual']} giriş özelliği bekleyecek şekilde ayarlı; gerçek geometri {diagnostic['expected']} özellik sağlıyor.",
            f"The head is configured for {diagnostic['actual']} input features; actual geometry supplies {diagnostic['expected']}.",
        )
        change = _bilingual(
            "in_features değerini Core tarafından hesaplanan beklenen sayıya eşitleyin.",
            "Set in_features to the expected count computed by Core.",
        )
    elif "flatten is supported only as the final schema2 feature block" in message:
        title = _bilingual(
            "Flatten konumu desteklenmiyor", "Unsupported Flatten position"
        )
        observed = _bilingual(
            "Flatten yalnız schema2 modelinin son özellik bloğu konumunda desteklenir.",
            "Flatten is supported only as the final schema2 feature block.",
        )
        change = _bilingual(
            "Flatten katmanını schema2 özellik zincirinin sonuna taşıyın ve açık başlık kullanın.",
            "Move Flatten to the end of the schema2 feature chain and use an explicit head.",
        )
    elif diagnostic.get("code") == "CHANNEL_MISMATCH":
        title = _bilingual(
            "Giriş kanal sözleşmesi uyuşmuyor", "Input channel contract mismatch"
        )
        change = _bilingual(
            "Katmanın kanal ayarını önceki katmanın Core tarafından doğrulanan çıkışıyla eşleştirin.",
            "Match the layer's channel setting to the previous layer's Core-validated output.",
        )
    reachable = {node_id} if node_id else set()
    for _ in report.get("nodes", {}):
        reachable.update(
            edge["target"]
            for edge in report.get("edges", [])
            if edge["source"] in reachable
        )
    blocked = sorted(
        key
        for key in reachable - {node_id}
        if report.get("nodes", {}).get(key, {}).get("status") == "blocked"
    )
    return dict(
        title=title,
        observed_condition=observed,
        recommended_change=change,
        blocked_nodes=blocked,
        diagnostic_context=dict(
            component=name,
            parameters=node.get("params", {}),
            input_shapes=report.get("nodes", {})
            .get(node_id, {})
            .get("input_shapes", []),
            expected=diagnostic.get("expected"),
            actual=diagnostic.get("actual"),
        ),
    )


def _conditions(record):
    p = record["payload"]
    provenance = record["provenance"]
    settings = p.get("prepared_effective", {})
    return dict(
        dataset=provenance.get("dataset"),
        split=provenance.get("split"),
        device=provenance.get("device"),
        data_sha256=provenance.get("data_sha256"),
        kind=record["kind"],
        head_mapping=provenance.get("head_mapping"),
        hardware=p.get("environment"),
        benchmark=p.get("configuration"),
        input=p.get("input"),
        training=dict(
            effective=settings, requested=p.get("settings", {}).get("requested")
        )
        if settings
        else None,
        evaluation=dict(
            configuration=p.get("configuration"),
            target_sha256=_digest(p["raw"]["targets"]),
            prediction_type=p["raw"].get("prediction_type"),
        )
        if p.get("format") == "hyso.evaluation.classification"
        else None,
    )


def _comparability(records):
    """No comparison claim is based on missing values or source declarations."""
    rows = [_conditions(row) for row in records]
    relevant = {"dataset", "split", "device", "kind"}
    if any(r["kind"] == "performance" for r in rows):
        relevant.update(("hardware", "benchmark", "input"))
    if any(r["kind"] == "training" for r in rows):
        relevant.add("training")
    if any(r["kind"] == "evaluation" for r in rows):
        relevant.add("evaluation")
    relevant.add("head_mapping")
    if any(row.get("data_sha256") for row in rows):
        relevant.add("data_sha256")
    missing, mismatches = [], []
    for key in sorted(relevant):
        values = [row[key] for row in rows]
        if key == "head_mapping" and all(value is None for value in values):
            continue
        if any(value is None for value in values):
            missing.append(key)
        elif any(value != values[0] for value in values[1:]):
            mismatches.append(key)
    return dict(
        status="CONDITIONS_DIFFER"
        if mismatches
        else "INSUFFICIENT_EVIDENCE"
        if missing
        else "CONDITIONS_MATCH",
        missing=missing,
        mismatches=mismatches,
        source_authenticity="UNVERIFIED_SOURCE"
        if any(r["authenticity"] == "UNVERIFIED_SOURCE" for r in records)
        else "LOCAL_EXECUTION",
        basis=_bilingual(
            "Kayıtlı koşullar ve beyan edilen veri/split kimlikleri. İstatistiksel anlamlılık değerlendirilmedi.",
            "Recorded conditions and caller-declared dataset/split identities. Statistical significance is unevaluated.",
        ),
    )


def _recommendation(
    rule,
    record,
    observation,
    suggestion,
    *,
    status="HYPOTHESIS",
    priority="MEDIUM",
    details=None,
    affected=None,
):
    evidence = dict(
        record_id=record["id"],
        payload_sha256=record["payload_sha256"],
        config_sha256=record["provenance"]["config_sha256"],
        details=details or {},
    )
    identity = _digest(dict(rule=rule, evidence=evidence, affected=affected))
    return dict(
        id=identity,
        deduplication_id=identity,
        deduplication_key=identity,
        rule=rule,
        scope=record["kind"].upper(),
        category=status,
        status=status,
        validation_status=status,
        affected=affected or record["provenance"].get("run_id", record["id"]),
        affected_entities=dict(
            model=record["provenance"]["config_sha256"],
            run=record["provenance"].get("run_id"),
            test=affected,
        ),
        evidence=evidence,
        evidence_references=[record["id"], record["payload_sha256"]],
        observation=observation[1],
        observed_condition=_bilingual(*observation),
        cause="Observed association; causality requires a controlled experiment",
        recommended_change=_bilingual(*suggestion),
        action=None,
        candidate_config=None,
        expected_effect="Requires prospective validation",
        tradeoffs=_bilingual(
            "Deney ek süre ve kaynak kullanır. Fayda ölçülmedi.",
            "An experiment uses additional time and resources. Benefit is unmeasured.",
        ),
        risk=_bilingual(
            "Sonuç ve model davranışı değişebilir. Açık kullanıcı deneyi gerekir.",
            "Results and model behavior may change. An explicit user experiment is required.",
        ),
        cost=_bilingual("Yeni kontrollü deney", "A new controlled experiment"),
        validation_method=_bilingual(
            "Kaynak kaydı doğrulandı. Önerilen değişiklik denenmedi.",
            "The source record is validated. The proposed change is untested.",
        ),
        confidence_basis=_bilingual(
            "Kayıtlı sayısal koşul ve açık eşik. Nedensellik kanıtı yok.",
            "Recorded numeric condition and explicit threshold. No causal proof.",
        ),
        priority=priority,
        requires_user=True,
    )


def _series(history, key):
    values = history.get(key, [])
    return (
        values
        if type(values) is list
        and len(values) >= 4
        and all(type(v) in (int, float) and math.isfinite(v) for v in values)
        else []
    )


def _diagnose(records):
    items = []
    for record in records:
        p = record["payload"]
        if p.get("format") == "hyso.performance.gradient_diagnostics":
            summaries = [
                row.get("preclip", {}).get("summary", {})
                for row in p.get("observations", [])
            ]
            if any(row.get("finite") is False for row in summaries):
                items.append(
                    _recommendation(
                        "NONFINITE_GRADIENT",
                        record,
                        (
                            "Native gradient kaydında sonlu olmayan türev gözlendi.",
                            "The native gradient record observed nonfinite gradients.",
                        ),
                        (
                            "İlgili güncelleme, loss ve scaler kaydını birlikte inceleyin.",
                            "Inspect the associated update, loss and scaler records together.",
                        ),
                        priority="HIGH",
                        details=dict(summary=p.get("summary")),
                    )
                )
            if len(summaries) >= 3 and all(
                row.get("finite") is True
                and row.get("global_l2", {}).get("status") == "measured"
                and row["global_l2"].get("value") == 0
                for row in summaries[-3:]
            ):
                items.append(
                    _recommendation(
                        "ZERO_GRADIENT_WINDOWS",
                        record,
                        (
                            "Son üç ölçülen güncelleme penceresinde gradient L2 normu sıfır.",
                            "Gradient L2 norm is zero in the last three measured update windows.",
                        ),
                        (
                            "Loss bağlantısını, dondurulmuş parametreleri ve veri örneklerini inceleyin.",
                            "Inspect loss connectivity, frozen parameters and data samples.",
                        ),
                        details=dict(last_windows=summaries[-3:]),
                    )
                )
        if record["kind"] == "training":
            h = p.get("history", {})
            train, val = _series(h, "train_loss"), _series(h, "val_loss")
            if train:
                details = dict(epochs=len(train), train_loss=train, val_loss=val)
                if (
                    val
                    and len(val) == len(train)
                    and train[-1] < train[0]
                    and val[-1] > val[0]
                    and val[-1] > train[-1]
                ):
                    items.append(
                        _recommendation(
                            "TRAIN_VALIDATION_DIVERGENCE",
                            record,
                            (
                                "Eğitim kaybı düşerken validation kaybı arttı.",
                                "Training loss fell while validation loss rose.",
                            ),
                            (
                                "Veri/split tutarlılığını inceleyin ve regularizasyonu ayrı deneyde değerlendirin.",
                                "Inspect dataset/split consistency and evaluate regularization in a separate experiment.",
                            ),
                            details=details,
                        )
                    )
                if (
                    max(train[-3:]) - min(train[-3:])
                    <= max(abs(train[-3]), 1e-12) * 0.01
                ):
                    items.append(
                        _recommendation(
                            "LOSS_PLATEAU",
                            record,
                            (
                                "Son üç epoch kayıp aralığı yüzde bir eşiğinin altında.",
                                "The last three loss values span at most the one-percent threshold.",
                            ),
                            (
                                "Learning-rate kaydını ve gradient kanıtını inceleyip tek değişkenli deney tasarlayın.",
                                "Inspect learning-rate and gradient evidence, then design a one-variable experiment.",
                            ),
                            details=dict(details, relative_span_threshold=0.01),
                        )
                    )
                if train[-1] > train[0] * 1.2:
                    items.append(
                        _recommendation(
                            "LOSS_INCREASE",
                            record,
                            (
                                "Son eğitim kaybı başlangıcın 1.2 katını aştı.",
                                "Final training loss exceeds 1.2 times the initial loss.",
                            ),
                            (
                                "Etiket/loss eşlemesini ve learning-rate planını inceleyin.",
                                "Inspect target/loss mapping and the learning-rate schedule.",
                            ),
                            priority="HIGH",
                            details=details,
                        )
                    )
                accuracy = _series(h, "train_acc")
                if (
                    accuracy
                    and max(accuracy) - min(accuracy) <= 1
                    and train[-1] >= train[0] * 0.99
                ):
                    items.append(
                        _recommendation(
                            "POSSIBLE_UNDERFITTING",
                            record,
                            (
                                "Kayıp azalmadı ve eğitim doğruluğu en çok bir yüzde puan değişti.",
                                "Loss did not decrease and training accuracy changed by at most one percentage point.",
                            ),
                            (
                                "Hedefleri, veri çeşitliliğini ve model kapasitesini kontrollü deneyle inceleyin.",
                                "Inspect targets, data variation and model capacity in a controlled experiment.",
                            ),
                            details=dict(details, train_accuracy=accuracy),
                        )
                    )
            rates = [
                row.get("learning_rate")
                for row in p.get("epoch_events", [])
                if row.get("learning_rate")
            ]
            if len(rates) >= 2 and rates[-1] != rates[0]:
                items.append(
                    _recommendation(
                        "LEARNING_RATE_CHANGED",
                        record,
                        (
                            "Kayıtlı learning-rate grupları eğitim sırasında değişti.",
                            "Recorded learning-rate groups changed during training.",
                        ),
                        (
                            "Değişimi istenen scheduler planıyla karşılaştırın.",
                            "Compare the change with the requested scheduler schedule.",
                        ),
                        status="INSUFFICIENT_EVIDENCE",
                        details=dict(rates=rates),
                    )
                )
        if record["kind"] == "testing":
            for case in p.get("cases", []):
                if case["status"] == "FAIL":
                    items.append(
                        _recommendation(
                            "TEST_FAILURE",
                            record,
                            (
                                "Kaynak test kaydı başarısız test içeriyor.",
                                "The source test record contains a failed test.",
                            ),
                            (
                                "Kaynak test mesajını ve ilgili alt sistemi inceleyin. Değişiklik nedenselliği ayrıca kanıt gerektirir.",
                                "Inspect the source test message and affected subsystem. Change causality requires additional evidence.",
                            ),
                            priority="HIGH",
                            affected=case["name"],
                            details=dict(
                                case=case,
                                expected="PASS",
                                actual="FAIL",
                                change_relation="UNAVAILABLE",
                            ),
                        )
                    )
        if (
            p.get("benchmark_type") == "inference"
            and p.get("configuration", {}).get("warmup_iterations") == 0
        ):
            items.append(
                _recommendation(
                    "BENCHMARK_NO_WARMUP",
                    record,
                    (
                        "Benchmark kaydı sıfır warmup iterasyonu kullanıyor.",
                        "The benchmark record uses zero warmup iterations.",
                    ),
                    (
                        "Soğuk ve kararlı çalışma etkilerini ayrı, açıkça başlatılan kısa deneyle ölçün.",
                        "Measure cold and steady-state effects in a separate explicitly started short experiment.",
                    ),
                    details=dict(
                        configuration=p["configuration"],
                        environment=p.get("environment"),
                    ),
                )
            )
    # Regressions are assessed only within identical architecture and known conditions.
    ordered = sorted(
        records, key=lambda r: ((r["provenance"].get("observed_at") or ""), r["id"])
    )
    tests = [
        r
        for r in ordered
        if r["kind"] == "testing" and r["provenance"].get("observed_at")
    ]
    for before, after in zip(tests, tests[1:]):
        if (
            before["provenance"]["config_sha256"]
            != after["provenance"]["config_sha256"]
        ):
            continue
        if _comparability([before, after])["status"] != "CONDITIONS_MATCH":
            continue
        previous = {c["name"]: c for c in before["payload"]["cases"]}
        for case in after["payload"]["cases"]:
            if (
                case["status"] == "FAIL"
                and previous.get(case["name"], {}).get("status") == "PASS"
            ):
                items.append(
                    _recommendation(
                        "TEST_REGRESSION",
                        after,
                        (
                            "Aynı test önceki kayıtta PASS, sonraki kayıtta FAIL.",
                            "The same test changed from PASS to FAIL in the later record.",
                        ),
                        (
                            "İki kaynak kaydı ve ilgili değişiklikleri birlikte inceleyin.",
                            "Review both source records and the associated changes.",
                        ),
                        priority="HIGH",
                        affected=case["name"],
                        details=dict(
                            previous_record_id=before["id"],
                            previous_sha256=before["payload_sha256"],
                            expected="PASS",
                            actual="FAIL",
                        ),
                    )
                )
    performance = [
        r
        for r in ordered
        if r["payload"].get("benchmark_type") == "inference"
        and r["provenance"].get("observed_at")
    ]
    for before, after in zip(performance, performance[1:]):
        gate = _comparability([before, after])
        baseline = before["payload"].get("stats", {}).get("pooled", {}).get("mean_ms")
        candidate = after["payload"].get("stats", {}).get("pooled", {}).get("mean_ms")
        if (
            gate["status"] == "CONDITIONS_MATCH"
            and type(baseline) in (int, float)
            and type(candidate) in (int, float)
            and baseline > 0
            and candidate > baseline * 1.1
        ):
            items.append(
                _recommendation(
                    "MEASURED_LATENCY_INCREASE",
                    after,
                    (
                        "Son kaydın ortalama gecikmesi önceki kaydın 1.1 katını aştı.",
                        "The later record's mean latency exceeds 1.1 times the earlier record.",
                    ),
                    (
                        "Ölçüm dağılımlarını ve donanım yükünü eş koşullu tekrar deneyinde inceleyin.",
                        "Inspect sample distributions and hardware load in a matched-condition repeat experiment.",
                    ),
                    priority="HIGH",
                    details=dict(
                        previous_record_id=before["id"],
                        previous_sha256=before["payload_sha256"],
                        baseline_ms=baseline,
                        candidate_ms=candidate,
                        comparability=gate,
                        statistical_significance="UNAVAILABLE",
                    ),
                )
            )
    return items
