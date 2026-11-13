"""Portable HTML and Markdown rendering from the common presentation contract."""
from __future__ import annotations

import html
import io
import json
import math
import textwrap


_WORDS = {
    "tr": dict(levels=dict(beginner="Başlangıç", technical="Teknik", expert="Uzman"),
        metrics="Metrikler", metric="Metrik", value="Değer", unit="Birim", status="Durum", scope="Ölçüm kapsamı",
        limits="Yorumlama sınırları", charts="Grafikler", features="P01–P35 kanıt durumu", advisor="Advisor açıklamaları",
        raw="Ham kaynak ve protokol", unknown="Kaydedilmemiş", empty="Bu bölüm için sayısal veri sağlanmadı.",
        presentation="Kayıtlı sonuçların sunumu", definition="Açıklama", guidance="Performans önerilerini incele",
        chart_data="Grafiğin kaynak verisi JSON çıktısında korunur.", epoch="Epoch", count="Örnek sayısı",
        experiment="Deney geçmişi", artifacts="Artifact ilişkileri", reason="Açıklama", percent="Göreli değişim (%)",
        pp="Yüzde puan farkı", execution="Ortak yürütücü durumları", protocol="Protokol ve kaynak sınırları"),
    "en": dict(levels=dict(beginner="Beginner", technical="Technical", expert="Expert"),
        metrics="Metrics", metric="Metric", value="Value", unit="Unit", status="Status", scope="Measurement scope",
        limits="Interpretation limits", charts="Charts", features="P01–P35 evidence status", advisor="Advisor explanations",
        raw="Raw source and protocol", unknown="Not recorded", empty="No numerical data was supplied for this section.",
        presentation="Presentation of retained results", definition="Explanation", guidance="Review performance suggestions",
        chart_data="Chart source data is retained in the JSON export.", epoch="Epoch", count="Sample count",
        experiment="Experiment history", artifacts="Artifact relationships", reason="Explanation", percent="Relative change (%)",
        pp="Percentage-point difference", execution="Common executor statuses", protocol="Protocol and source limits"),
}
_STATUS = {
    "missing": ("Eksik", "Missing"), "partial": ("Kısmi kanıt", "Partial evidence"),
    "validated": ("Native iç tutarlılık doğrulandı", "Native consistency validated"),
    "unsupported": ("Desteklenmiyor", "Unsupported"), "not_run": ("Çalıştırılmadı", "Not run"),
    "invalid": ("Geçersiz", "Invalid"), "unknown": ("Bilinmiyor", "Unknown"),
    "measured": ("Ölçülmüş", "Measured"), "derived": ("Kaynak veriden türetilmiş", "Derived from source data"),
    "estimated": ("Tahmini", "Estimated"), "recorded": ("Kayıtlı", "Recorded"),
    "computed": ("Kayıtlı yapıdan hesaplandı", "Computed from retained structure"),
    "insufficient_evidence": ("Yetersiz kanıt", "Insufficient evidence"),
    "created": ("Oluşturuldu", "Created"), "running": ("Çalışıyor", "Running"),
    "completed": ("Tamamlandı", "Completed"), "failed": ("Başarısız", "Failed"),
    "interrupted": ("Kesintiye uğradı", "Interrupted"),
    "COMPARABLE": ("Karşılaştırılabilir", "Comparable"), "INCOMPATIBLE": ("Uyumsuz", "Incompatible"),
    "INSUFFICIENT_EVIDENCE": ("Yetersiz kanıt", "Insufficient evidence"),
    "NOT_SUPPORTED_IN_COMPARISON": ("Karşılaştırma desteği yok", "Unsupported in comparison"),
    "COMPLETED": ("Tamamlandı", "Completed"), "FAILED": ("Başarısız", "Failed"),
    "SKIPPED": ("Atlandı", "Skipped"), "UNSUPPORTED": ("Desteklenmiyor", "Unsupported"),
    "SUPPORTED": ("Yürütücü desteği var", "Executor supported"),
    "MISSING_PREREQUISITE": ("Önkoşul eksik", "Missing prerequisite"),
    "NOT_IMPLEMENTED_IN_EXECUTOR": ("Ortak yürütücüye bağlı değil", "Not connected to common executor"),
    "verified": ("Dosya bütünlüğü doğrulandı", "File integrity verified"),
    "mismatch": ("Dosya içeriği değişmiş", "File content mismatch"),
    "unavailable": ("Erişim bilgisi yok", "Unavailable"), "unreadable": ("Okunamıyor", "Unreadable"),
    "invalid_result": ("Sonuç doğrulaması başarısız", "Result validation failed"),
}


def _status(value, language):
    translated = _STATUS.get(value)
    return value if translated is None else translated[0 if language == "tr" else 1] + " [" + value + "]"


def _string(value, words):
    if value is None:
        return words["unknown"]
    if type(value) in (dict, list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _escape(value, words):
    return html.escape(_string(value, words), quote=True)


def _md(value, words):
    return _escape(value, words).replace("|", "&#124;").replace("\n", " ").replace("`", "&#96;").replace("[", "&#91;").replace("]", "&#93;")


def _table(headers, rows, words):
    return "<div class='table-scroll'><table><thead><tr>" + "".join("<th>" + _escape(h, words) + "</th>" for h in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + _escape(cell, words) + "</td>" for cell in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def _metric_rows(source, words, language):
    return [[m["label"], m["value"], m["unit"], _status(m["status"], language), m["domain"]] for m in source["metrics"]]


def _fragment(link):
    from .report import ReportError
    if type(link) is not str or not link.startswith("#"):
        raise ReportError("local_recommendation_link_required")
    return html.escape(link, quote=True)


def _svg(chart, words):
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.ticker import MaxNLocator
    from .report import ReportError

    if chart["kind"] not in ("line", "bar", "histogram") or not chart["series"]:
        raise ReportError("invalid_chart_kind_or_series")
    for series in chart["series"]:
        values = series["y"]
        if not values or any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)) for value in values):
            raise ReportError("invalid_chart_values")
        if not any(value is not None for value in values):
            raise ReportError("chart_has_no_observations")
    with rc_context({"svg.hashsalt": "hyso-report-v1", "svg.fonttype": "none", "font.family": "DejaVu Sans", "text.parse_math": False}):
        height = max(4.5, .55 * len(chart["series"])) if chart["kind"] == "bar" else 4.5
        figure = Figure(figsize=(9, height), layout="constrained", facecolor="#ffffff")
        FigureCanvasSVG(figure)
        axis = figure.subplots()
        colors = ("#087f8c", "#d96c32", "#4856a8")
        if chart["kind"] == "histogram":
            for i, series in enumerate(chart["series"]):
                axis.hist(series["y"], bins=20, alpha=.6, label=series["label"], color=colors[i % len(colors)])
            axis.set_xlabel(chart["unit"] or words["unknown"])
            axis.set_ylabel(words["count"])
        elif chart["kind"] == "line":
            for i, series in enumerate(chart["series"]):
                if len(series["x"]) != len(series["y"]) or any(type(x) not in (int, float) or not math.isfinite(x) for x in series["x"]):
                    raise ReportError("invalid_chart_axis")
                # NaN preserves unobserved gaps in a curve.
                axis.plot(series["x"], [float("nan") if v is None else v for v in series["y"]],
                          marker="o", markersize=3, label=series["label"], color=colors[i % len(colors)])
            axis.set_xlabel(words["epoch"])
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set_ylabel(chart["unit"] or words["unknown"])
        else:
            for i, series in enumerate(chart["series"]):
                if len(series["y"]) != 1 or series["y"][0] is None:
                    raise ReportError("invalid_bar_observation")
                axis.barh(i, series["y"][0], color=colors[i % len(colors)])
            axis.set_yticks(range(len(chart["series"])), [textwrap.fill(s["label"], 34) for s in chart["series"]], fontsize=9)
            axis.set_xlabel(chart["unit"] or words["unknown"])
            axis.invert_yaxis()
        if chart["kind"] in ("line", "histogram"):
            axis.legend(frameon=False, fontsize=9)
        axis.set_title(chart["title"], loc="left", pad=16, fontsize=12, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.15, axis="y")
        axis.set_axisbelow(True)
        output = io.StringIO()
        figure.savefig(output, format="svg", metadata={"Date": None, "Creator": "HYSo"})
        raw = output.getvalue()
        return raw[raw.index("<svg"):]


def _actions(source):
    view = source["advisor"]
    if view is None:
        return []
    if "text" in view:
        return view["text"]["actions"]
    return [action for side in view.get("advisor_observations", {}).values() for action in side["text"]["actions"]]


def _recommendations(source, words, language, *, as_html):
    """Render manual cards, including exact retained evidence at every level."""
    view = source.get("recommendations")
    if view is None:
        return "" if as_html else []
    tr = language == "tr"
    labels = dict(detected="Tespit edilen durum" if tr else "Detected condition",
                  change="Önerilen değişiklik" if tr else "Suggested change",
                  why="Neden faydalı olabilir?" if tr else "Why it may help",
                  risks="Olası yan etkiler" if tr else "Possible side effects",
                  steps="Uygulama adımları" if tr else "Manual steps",
                  verification="Yeniden ölçme ve doğrulama" if tr else "Remeasurement and verification",
                  evidence="Kaynak kanıtları" if tr else "Source evidence",
                  missing="Eksik veya uygun olmayan kanıt" if tr else "Missing or ineligible evidence",
                  gates="Engelleyen koşullar" if tr else "Blocking conditions",
                  original="Mevcut kural ve plan" if tr else "Existing rule and plan")
    states = dict(evidence_review="Kanıt incelemesi" if tr else "Evidence review",
                  measurement_request="Ölçüm ihtiyacı" if tr else "Measurement needed",
                  conditional_experiment="Koşullu deney adayı" if tr else "Conditional experiment candidate")
    summary = ("Manuel inceleme. Değişiklik uygulanmadı, ölçüm başlatılmadı. Kazanç ve öncelik sırası belirlenmedi."
               if tr else "Manual review. No change applied or measurement started. Benefit and priority ranking are not established.")
    parts = ["<p>" + summary + "</p>"] if as_html else ["", summary, ""]
    records = {(s["side"], r["record_id"]): r for s in view["sources"] for r in s["report"]["index"]["records"]}

    def paragraph(label, value):
        if as_html:
            parts.append("<p><strong>" + label + "</strong> " + _escape(value, words) + "</p>")
        else:
            parts.extend(["", "**" + label + "** " + _md(value, words), ""])

    for area in view["areas"]:
        cards = [c for c in view["recommendations"] if c["area"] == area["area"]]
        if not cards:
            continue
        parts.append("<h3>" + _escape(area["label"], words) + "</h3>" if as_html else "### " + _md(area["label"], words))
        for card in cards:
            # Presentation JSON is inspectable, not an authenticity certificate.
            # Restrict anchors before using them in HTML or Markdown attributes.
            card = dict(card)
            for key in ("recommendation_anchor", "evidence_anchor", "verification_anchor"):
                value = card[key]
                if type(value) is not str or not value or len(value) > 256 or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in value):
                    from .report import ReportError
                    raise ReportError("invalid_recommendation_anchor")
            anchor = card["recommendation_anchor"]
            if as_html:
                parts.append("<article id='" + anchor + "'><h4>" + _escape(card["title"], words) + "</h4>")
            else:
                parts.extend(["", '<a id="' + anchor + '"></a>', "", "#### " + _md(card["title"], words), ""])
            paragraph(words["status"], states[card["action_kind"]] + " · " + card["side"])
            for key in ("detected", "change", "why", "risks"):
                paragraph(labels[key], card[key])
            if as_html:
                parts.append("<p><a href='#" + card["evidence_anchor"] + "'>" + labels["evidence"] + "</a> · <a href='#" +
                             card["verification_anchor"] + "'>" + labels["verification"] + "</a></p>")
            else:
                parts.extend(["[" + labels["evidence"] + "](#" + card["evidence_anchor"] + ") · [" +
                              labels["verification"] + "](#" + card["verification_anchor"] + ")", ""])
            for key in ("steps", "verification"):
                aid = card["verification_anchor"] if key == "verification" else anchor + "-steps"
                if as_html:
                    parts.extend(["<div id='" + aid + "'><h5>" + labels[key] + "</h5><ol>",
                                  *["<li>" + _escape(step, words) + "</li>" for step in card[key]], "</ol></div>"])
                else:
                    parts.extend(['<a id="' + aid + '"></a>', "", "**" + labels[key] + "**", "",
                                  *[str(i) + ". " + _md(step, words) for i, step in enumerate(card[key], 1)], ""])
            if card["blocked_by"]:
                paragraph(labels["gates"], card["blocked_by"])
            paragraph(labels["missing"], card["missing_evidence"])
            evidence = dict(references=card["evidence"], coverage=card["coverage"],
                            records=[records[(card["side"], rid)] for rid in card["retained_record_ids"]])
            if as_html:
                parts.append("<details id='" + card["evidence_anchor"] + "'><summary>" + labels["evidence"] + "</summary>")
            else:
                parts.extend(['<a id="' + card["evidence_anchor"] + '"></a>', "", "**" + labels["evidence"] + "**", ""])
            for metric in card["observations"]:
                paragraph(metric["label"], str(metric["value"]) + " " + metric["reference"]["unit"])
                paragraph(words["definition"], metric["description"])
            raw = json.dumps(evidence, ensure_ascii=False, indent=2)
            if as_html:
                parts.extend(["<pre>" + _escape(raw, words) + "</pre></details>",
                              "<details><summary>" + labels["original"] + "</summary><pre>" +
                              _escape(json.dumps(dict(rule_id=card["rule_id"], rule_version=card["rule_version"],
                                                     finding=card["original_finding"], plan=card["plan"]), ensure_ascii=False, indent=2), words) + "</pre></details></article>"])
            else:
                parts.extend(["```json", raw.replace("`", "\\u0060"), "```", ""])
                paragraph(labels["original"], dict(rule_id=card["rule_id"], rule_version=card["rule_version"],
                                                  finding=card["original_finding"], plan=card["plan"]))
    if view["comparison"] is not None:
        paragraph("Comparability", view["comparison"]["metrics"])
    return "".join(parts) if as_html else parts


def _experiment_html(experiment, words, language, level):
    if experiment is None:
        return ""
    content = ["<section><h2>" + words["experiment"] + "</h2>",
               "<p>" + _escape(experiment["experiment"]["name"], words) + " · " + _escape(_status(experiment["status"], language), words) + "</p>"]
    rows = [[r["run"]["id"], r["run"]["kind"], _status(r["run"]["status"], language), r["run"]["metric_count"]] for r in experiment["runs"]]
    content.append(_table(["Run", "Kind", words["status"], words["metrics"]], rows, words))
    for run in experiment["runs"]:
        if run["metrics"]:
            rows = [[r["name"], r["value"], r["unit"], r["axis_kind"], r["axis_index"], r["source"]] for r in run["metrics"]]
            content += ["<h3>" + words["metrics"] + " · " + _escape(run["run"]["id"], words) + "</h3>",
                        _table([words["metric"], words["value"], words["unit"], "Axis", "Index", "Source"], rows, words)]
        rows = [[r["artifact"]["id"], r["roles"], _status(r.get("result_status", r["availability"]), language), r.get("result_reason", r["reason"])] for r in run["artifacts"]]
        if rows:
            content += ["<h3>" + words["artifacts"] + " · " + _escape(run["run"]["id"], words) + "</h3>",
                        _table(["Artifact", "Role", words["status"], words["reason"]], rows, words)]
    if level == "expert":
        content += ["<details><summary>" + words["raw"] + "</summary><pre>" + _escape(json.dumps(experiment, ensure_ascii=False, indent=2), words) + "</pre></details>"]
    content.append("</section>")
    return "".join(content)


def html_report(data):
    language, level = data["language"], data["level"]
    words = _WORDS[language]
    catalog = {row["feature_id"]: row for row in data["catalog"]}
    css = """
    :root{color-scheme:light;color:#182b3a;background:#eff3f6;font-family:Inter,'Segoe UI',sans-serif}
    *,*::before,*::after{box-sizing:border-box}
    body{margin:0;line-height:1.6;overflow-wrap:anywhere}main{width:100%;max-width:1180px;margin:auto;padding:42px 24px}
    header{border-bottom:3px solid #087f8c;margin-bottom:28px;padding-bottom:22px}h1{font-size:2.5rem;margin:.2em 0}
    h2{font-size:1.65rem}h3{font-size:1.1rem}section{background:white;padding:28px;margin:24px 0;border-radius:10px}
    a{color:#086b81}nav{display:flex;gap:18px;flex-wrap:wrap}.table-scroll{max-width:100%;overflow-x:auto}
    table{width:100%;min-width:640px;border-collapse:collapse;font-size:.88rem}th,td{text-align:left;padding:12px;border-bottom:1px solid #dfe7ed;vertical-align:top;overflow-wrap:anywhere}
    td:nth-child(2){white-space:nowrap}
    th{background:#eef5f7}figure{margin:28px 0;max-width:100%;overflow-x:auto}
    svg{display:block;min-width:864px;max-width:100%;height:auto}svg text{font-family:Arial,'Segoe UI',sans-serif!important}
    figcaption{color:#52616d;font-size:.9rem}
    details{padding:12px 0;border-bottom:1px solid #dfe7ed}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem;background:#f4f6f8;padding:14px}
    .metadata{color:#52616d}.metric-note{border-left:3px solid #91b9c2;padding-left:16px;margin:20px 0}
    @media print{body{background:white}main{max-width:none;padding:0}section{break-inside:auto}figure{break-inside:avoid}svg{min-width:0}nav{display:none}}
    """
    parts = ["<!doctype html><html lang='" + language + "'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>" + _escape(data["title"], words) + "</title><style>" + css + "</style></head><body><main>",
             "<header><p class='metadata'>" + words["presentation"] + " · " + words["levels"][level] + "</p><h1>" + _escape(data["title"], words) + "</h1><nav>"]
    parts.extend("<a href='#source-" + _escape(source["source_id"], words) + "'>" + _escape(source["title"], words) + "</a>" for source in data["sources"])
    parts.append("</nav></header>")
    parts.append(_experiment_html(data["experiment"], words, language, level))
    if not data["sources"] and data["experiment"] is None:
        parts.append("<p>" + words["empty"] + "</p>")
    for source in data["sources"]:
        identity = _escape(source["source_id"], words)
        parts += ["<section id='source-" + identity + "'><h2>" + _escape(source["title"], words) + "</h2>"]
        if source["features"]:
            parts.append("<h3>" + words["features"] + "</h3>")
            for feature in source["features"]:
                info = catalog[feature["feature_id"]]
                parts += ["<details><summary>" + feature["feature_id"] + " · " + _escape(info["name"], words) + " · " + _escape(_status(feature["status"], language), words) + "</summary>",
                          "<p>" + _escape(info["explanations"][level], words) + "</p><p>" + _escape(feature["reason"], words) + "</p><p>" + _escape(info["limits"], words) + "</p></details>"]
        if source.get("execution"):
            rows = [[r["model_id"], r["feature_id"], _status(r["status"], language), r["reasons"]] for r in source["execution"]["features"]]
            parts += ["<h3>" + words["execution"] + "</h3>", _table(["Model", "P", words["status"], words["reason"]], rows, words)]
        if not source.get("recommendations"):
            parts.append("<h3>" + words["metrics"] + "</h3>")
        if source["metrics"]:
            parts.append(_table([words["metric"], words["value"], words["unit"], words["status"], "Domain"], _metric_rows(source, words, language), words))
        elif not source.get("recommendations"):
            parts.append("<p>" + words["empty"] + "</p>")
        for metric in source["metrics"]:
            parts += ["<details class='metric-note'><summary>" + _escape(metric["label"], words) + "</summary><p>" + _escape(metric["description"], words) + "</p>",
                      "<p><strong>" + words["scope"] + "</strong> " + _escape(metric["scope"], words) + "</p>",
                      "<p><strong>" + words["limits"] + "</strong> " + _escape(metric["limits"], words) + "</p>"]
            for key, label in (("percent_change", words["percent"]), ("percentage_point_change", words["pp"])):
                if key in metric:
                    parts.append("<p>" + label + " " + _escape(metric[key], words) + "</p>")
            parts.extend("<p><a href='" + _fragment(link) + "'>" + words["guidance"] + "</a></p>" for link in metric["recommendation_links"])
            parts.append("</details>")
        for chart in source["charts"]:
            parts += ["<figure>" + _svg(chart, words) + "<figcaption>" + _escape(chart["domain"], words) + " · " + words["chart_data"] + "</figcaption>",
                      "<details><summary>" + words["protocol"] + "</summary><pre>" + _escape(chart["scope"], words) + "</pre></details></figure>"]
        parts.append(_recommendations(source, words, language, as_html=True))
        actions = _actions(source)
        if actions:
            parts.append("<div id='advisor-" + identity + "'><h3>" + words["advisor"] + "</h3>")
            for action in actions:
                parts += ["<p>" + _escape(action.get("action"), words) + "</p><p>" + _escape(action.get("why"), words) + "</p>",
                          "<p>" + _escape(action.get("missing"), words) + "</p>"]
            parts.append("</div>")
        parts += ["<h3>" + words["limits"] + "</h3><ul>", *["<li>" + _escape(note, words) + "</li>" for note in source["limits"]], "</ul>"]
        if level == "expert":
            raw = dict(source=source["raw"], execution=source.get("execution"), advisor=source["advisor"])
            parts += ["<details><summary>" + words["raw"] + "</summary><pre>" + _escape(json.dumps(raw, ensure_ascii=False, indent=2), words) + "</pre></details>"]
        parts.append("</section>")
    parts += ["<footer class='metadata'>SHA256 " + _escape(data["content_sha256"], words) + "</footer></main></body></html>"]
    return "".join(parts)


def markdown(data):
    language = data["language"]
    words = _WORDS[language]
    catalog = {row["feature_id"]: row for row in data["catalog"]}
    lines = ["# " + _md(data["title"], words), "", words["presentation"] + " · " + words["levels"][data["level"]], ""]
    experiment = data["experiment"]
    if experiment:
        lines += ["## " + words["experiment"], "", _md(experiment["experiment"]["name"], words), "", _status(experiment["status"], language), ""]
        for detail in experiment["runs"]:
            run = detail["run"]
            lines += ["- " + run["id"] + " · " + run["kind"] + " · " + _status(run["status"], language)]
            for metric in detail["metrics"]:
                lines.append("  - " + _md(metric["name"], words) + " · " + _md(metric["value"], words) + " " + _md(metric["unit"], words) + " · " +
                             _md(metric["axis_kind"], words) + " " + str(metric["axis_index"]) + " · " + _md(metric["source"], words))
            for item in detail["artifacts"]:
                lines.append("  - " + item["artifact"]["id"] + " · " + _status(item.get("result_status", item["availability"]), language) + " · " + _md(item.get("result_reason", item["reason"]), words))
        lines.append("")
        if data["level"] == "expert":
            lines += ["```json", json.dumps(experiment, ensure_ascii=False, indent=2).replace("`", "\\u0060"), "```", ""]
    for source in data["sources"]:
        lines += ["## " + _md(source["title"], words), ""]
        if source["features"]:
            lines += ["| P | " + words["status"] + " | " + words["definition"] + " |", "|---|---|---|"]
            for feature in source["features"]:
                info = catalog[feature["feature_id"]]
                lines.append("| " + feature["feature_id"] + " | " + _status(feature["status"], language) + " | " + _md(info["explanations"][data["level"]] + " " + feature["reason"], words) + " |")
            lines.append("")
        if not source.get("recommendations"):
            lines += ["| " + " | ".join([words["metric"], words["value"], words["unit"], words["status"], "Domain"]) + " |", "|---|---:|---|---|---|"]
        for row in _metric_rows(source, words, language):
            lines.append("| " + " | ".join(_md(value, words) for value in row) + " |")
        for metric in source["metrics"]:
            lines += ["", "### " + _md(metric["label"], words), "", _md(metric["description"], words), "",
                      words["scope"] + " " + _md(metric["scope"], words), "", words["limits"] + " " + _md(metric["limits"], words)]
            if "percent_change" in metric:
                lines += ["", words["percent"] + " " + _md(metric["percent_change"], words), "",
                          words["pp"] + " " + _md(metric["percentage_point_change"], words)]
            for link in metric["recommendation_links"]:
                lines += ["", "[" + words["guidance"] + "](" + _fragment(link) + ")"]
        for chart in source["charts"]:
            lines += ["", "### " + _md(chart["title"], words), "", words["chart_data"], "", _md(chart["scope"], words)]
        lines.extend(_recommendations(source, words, language, as_html=False))
        actions = _actions(source)
        if actions:
            lines += ["", '<a id="advisor-' + source["source_id"] + '"></a>', "", "### " + words["advisor"], ""]
            lines.extend("- " + _md(action.get("action"), words) + " " + _md(action.get("why"), words) for action in actions)
        lines += ["", "### " + words["limits"], "", *["- " + _md(note, words) for note in source["limits"]], ""]
        if data["level"] == "expert":
            raw = dict(source=source["raw"], execution=source.get("execution"), advisor=source["advisor"])
            lines += ["### " + words["raw"], "", "```json", json.dumps(raw, ensure_ascii=False, indent=2).replace("`", "\\u0060"), "```", ""]
    lines += ["SHA256 " + data["content_sha256"], ""]
    return "\n".join(lines)
