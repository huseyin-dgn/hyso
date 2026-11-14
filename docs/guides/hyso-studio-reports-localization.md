> Historical document: the web product was retired in stage 13.5.6. Do not use the launch/API instructions below as an active product guide. See [Python-first transition](../reports/stage13-5-6-python-first-transition.md).

# Studio reports and localization

The interface has English and Turkish navigation, core editing labels, settings controls, report controls and Core bilingual diagnostics. Language is a GLOBAL preference persisted in Studio SQLite. Technical identifiers, model JSON keys, node IDs and status codes retain their original spelling. English is the fallback language. Settings contains an audit of missing UI keys and metadata fields using fallback.

Commercial component tutorials, parameter labels/descriptions, principal workflows, status labels and engineering prompts have Turkish and English text. Technical API identifiers, original source evidence and expandable native source details retain their original spelling. Changing language leaves configuration hashes and numeric evidence unchanged. Tests compare report values and provenance across languages and audit missing UI/metadata keys.

## Report selection

Choose language, JSON/HTML/Markdown output, display precision, optional run and experiment IDs, observed-date range, metric keys and sections. Architecture, diagnostics, retained records, recommendations and available charts have separate controls. Reports are generated through a persisted `report` job. Their measurement records always match the selected architecture's exact configuration SHA-256. A conflicting explicit model hash is rejected.

The model selector offers the current draft, saved draft revisions and immutable validated versions. Version reports additionally filter source version identity. Record-kind checkboxes select training, testing, performance and evaluation records. Choosing a saved report model leaves the active editor draft intact. Metrics are selected by readable labels from actual records. The API's optional `record_kinds` array provides the same filter.

Separate checkboxes select the provenance block and diagnostic warnings. The optional API fields `include_provenance` and `include_warnings` default to true for existing callers. Native source identities remain in retained technical evidence when the provenance presentation block is hidden. Record filters also constrain recommendation evidence.

Date filters apply to the caller's `observed_at` timestamp in UTC. Import time remains `imported_at` and supplies no replacement observation timestamp. Missing observation time excludes a record from an observed-date filter. Missing device, dataset, split or experiment information remains unavailable in the retained provenance.

`hyso.studio.report` schema1 contains the configuration hash, selected filters, source record IDs, payload SHA-256 values, declared provenance, source authenticity, native presentations and selected architecture/diagnostics. Native `PresentationReport` data comes from the existing `create_report` path. Filtered presentation digests are recalculated. JSON retains exact numeric values. Display precision applies to scalar metric values in the rendered HTML and Markdown presentations.

HTML downloads embed the existing HYSo renderer's metric tables and supported Matplotlib charts in sandboxed `srcdoc` frames. The basic Studio preview shows readable metrics and actual embedded chart images; full source evidence is expandable. Markdown uses the existing renderer. The Studio envelope retains provenance and selected architecture as escaped structured data. PDF and CSV export controls are unavailable. Targeted Chromium screenshots cover the Studio preview; this is not an audit of every exported report layout.

## Result imports

Native JSON uploads retain their original numeric literals in the HTTP payload. This preserves `1.0` values and large integer evidence required by native integrity checks. The evidence viewer uses a backend canonical payload string. Model identity follows Core-normalized configuration, exact integer/float equivalence and deterministic node ordering. Signed zero is preserved. Browser model editing uses BigInt and decimal text for lossless large integers. JSON export emits numeric tokens that retain their exact values.

The closed import readers support training-session results, classification evaluation, latency, throughput, memory, footprint, gradient diagnostics and existing inference BenchmarkResult JSON. Training restoration reuses the existing validated workflow reader. Native results reuse their existing consistency validators. External Nsight/file references, arbitrary Python and pickle input are unavailable.

Testing additionally accepts `hyso.studio.test_result` schema1 and bounded JUnit XML. JUnit uploads allow at most 64 KiB and 256 cases, reject DTD/entity declarations and retain their original uploaded XML plus SHA-256. PASS, FAIL, SKIP and pytest XFAIL states remain distinct.

Import asks the user to explicitly choose the current or a saved source model/version. The UI derives its exact hash; an external hash is available under technical details. A retained benchmark with embedded Core architecture must match this declaration. Associations without an embedded architecture remain `CALLER_DECLARED` and their authenticity remains `UNVERIFIED_SOURCE`. Native consistency validates retained values and derivations. Source authorship and claimed model relationship remain declarations.

The panels expose original payloads and native metric status. Native derived or recorded values retain their source-status label alongside the Studio MEASURED/ESTIMATED/UNAVAILABLE classification. Training observer time retains its existing scope limitations. It supplies no P19 or GPU timing claim.

External imports are `IMPORTED_UNVERIFIED` with `UNVERIFIED_SOURCE` authenticity. Results observed by explicit jobs in this Studio workspace are `MEASURED` with `LOCAL_EXECUTION`. Dataset/split associations still require explicit evidence or declaration. These states remain separate from native metric status and estimation labels.

## Türkçe rapor ve kayıt rehberi

Rapor ekranında güncel taslak veya kayıtlı model revision'ı seçin. Dil, JSON/HTML/Markdown, hassasiyet, run/experiment, UTC tarih aralığı, metrik anahtarları, kayıt türleri ve rapor bölümlerini ayarlayın. Rapor açık bir job ile oluşturulur. Her kayıt seçilen mimarinin config SHA-256 değeriyle eşleşmelidir. Boş değerler sıfıra çevrilmez. PDF ve CSV desteklenmez.

TR/EN raporları aynı sayısal kanıt ve provenance verisini kullanır. Hassasiyet render edilen scalar metrikleri biçimlendirir. JSON tam değerleri taşır. Tarih filtresi gözlem zamanına uygulanır. Eksik gözlem zamanı aktarım zamanıyla doldurulmaz. Grafikler mevcut native renderer veya kayıtlı epoch dizilerinden üretilir.

İçe aktarımda kaynak model hash, varsa run, gözlem zamanı, veri/split ve cihaz kimliğini açıkça belirtin. Native JSON sayı metni değiştirilmeden gönderilir. JUnit boyutu 64 KiB ve 256 case ile sınırlıdır. PASS/FAIL/SKIP/XFAIL ayrı kalır. Kaynak XML ve hash korunur.

Dış kaynaklar IMPORTED_UNVERIFIED olarak gösterilir. Yerel açık CPU işinde gözlenen kayıt MEASURED olur. Tahmini aktivasyon ve parametre değerleri ölçümle karıştırılmaz. Ham kaynak kimlikleri ve teknik anahtarlar çevrilmez. Güvenlik/ticari politika ayarları dil seçimiyle değişmez.
