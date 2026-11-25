"""Version 1 presentation dictionaries. Never translate canonical evidence."""
LOCALIZATION_VERSION = '1'

TEXT = {
    'tr': {
        'title': 'HYSo — Model Performans Raporu',
        'performance': 'Modelin mevcut performansı', 'goals': 'Kullanıcı hedeflerinin durumu',
        'findings': 'Önemli bulgular', 'actions': 'Önerilen sonraki incelemeler veya deneyler',
        'why': 'Neden bunlar öneriliyor?', 'missing': 'Eksik kanıtlar', 'risks': 'Riskler ve sınırlamalar',
        'execution': 'İşlem yapılıp yapılmadığı', 'technical': 'Teknik ayrıntılar',
        'source': 'Kaynak Advisor raporu', 'analysis': 'Kaynak analiz raporu',
        'no_goals': 'Kullanıcı sayısal hedef belirtmedi; yeni hedef veya sert sınır oluşturulmadı.',
        'no_metrics': 'Bu raporda desteklenen sayısal ölçüm yok. Eksik değerler sıfır kabul edilmedi.',
        'separate': 'Bu ölçümler ayrı kayıt ve protokollere aittir; birlikte doğrulanmış tek bir deployment koşulu sayılmaz.',
        'no_candidate': 'Mevcut kanıtlar henüz bir optimizasyon deneyi önermek için yeterli değil.',
        'candidate': 'Koşullu deney adayı', 'review': 'Sonraki kanıt incelemesi',
        'not_authorized': 'Yetkilendirilmedi; hiçbir işlem başlatılmayacak.',
        'not_executed': 'Bu raporlama sırasında model üzerinde herhangi bir değişiklik veya yeni çalışma gerçekleştirilmedi.',
        'authority': 'Bu sunum yalnız görüntüleme içindir; kanonik AdvisorReport yerine geçmez ve kanıt doğrulamasında otorite değildir.',
        'original': 'Özgün teknik kayıt — serbest kaynak metni çevrilmedi.',
        'unknown_rule': 'Bu kural veya sürümü için güvenilir yerelleştirme yok; özgün teknik kayda bakın.',
        'unknown_metric': 'Bu bulgunun desteklenen typed metrik açıklaması yok; sayısal anlam tahmin edilmedi.',
        'partial': 'Bazı kayıt ilişkileri kullanıcı beyanı veya eksik bilgidir; native olarak doğrulanmış ilişkiye yükseltilmedi.',
        'local_goal': 'Hedef sonucu yalnız ilgili kaydın kapsamındaki gözlemdir; deployment kabulü veya gelecekte başarı garantisi değildir.',
        'future': 'Gelecekte gerçekleştirilebilirlik', 'support': 'Kanıt desteği', 'input': 'İstek geçerliliği',
        'observed': 'Kayıt kapsamındaki sonuç', 'value': 'Mevcut ölçülen değer', 'target': 'Hedef',
        'consistency': 'Hedefler arası tutarlılık', 'hard': 'sert koşul', 'soft': 'tercih',
        'no_actions': 'Bu raporda ek inceleme veya deney planı yok; boşluk sahte öneriyle doldurulmadı.',
        'change': 'Değişecek değişken', 'fixed': 'Sabit koşullar', 'verify': 'Sonuç nasıl doğrulanacak?',
        'accept': 'Kabul koşulları', 'unknown_effect': 'Beklenen kazanç ve maliyet bilinmiyor; hızlanma veya başarı yüzdesi verilmedi.',
        'experiment_fixed': 'Yalnız batch boyutu değişebilir. Model, cihaz, precision, backend, veri işleme ve zamanlama yöntemi sabit tutulmalıdır.',
        'experiment_verify': 'Her seçilen batch için tamamlanan görüntü hızı, P95 gecikme, süreç RAM kullanımı, bağlı kalite değerlendirmesi ve sınırlı kararlılık yeniden kontrol edilmelidir.',
        'experiment_accept': 'Tüm açık sert koşullar bağlı kanıtlarla karşılanmadan genel başarı ilan edilemez. Bağımsız doğrulama ve ComparisonContract hâlâ gereklidir.',
        'experiment_missing': 'Açık deney bütçesi, sınırlı batch noktaları, çalışma kabiliyeti ve bağlı bellek/kalite kabul kanıtları tamamlanmalıdır.',
        'experiment_risks': 'Batch büyümesi gecikmeyi, RAM kullanımını ve bellek yetersizliği riskini artırabilir; doğruluk ve kararlılık etkileri bilinmiyor.',
        'bindings': 'native: üretici kaydından; declared: kullanıcı beyanı; missing: eksik. Bu etiketler birbirine dönüştürülmez.',
        'technical_note': 'Aşağıdaki kayıt kimlik, metrik yolu, scope/protocol, eligibility, provenance, rule version ve kabul sınırlarını eksiksiz korur. Kaynak ifadeler özgün dilindedir.',
        'counts': 'Bulgu sayıları', 'unknown_cost': 'Bütçe, maliyet veya çalışma izni bu sunumdan türetilemez.',
        'no_risk_translation': 'Ek kaynak riskleri özgün teknik kayıtta korunuyor; çevrilerek yeni bir güvenlik kararı üretilmedi.',
    },
    'en': {
        'title': 'HYSo — Model Performance Report',
        'performance': 'Current model performance', 'goals': 'User goal status', 'findings': 'Important findings',
        'actions': 'Suggested next reviews or experiments', 'why': 'Why these are suggested', 'missing': 'Missing evidence',
        'risks': 'Risks and limitations', 'execution': 'Whether any action was performed', 'technical': 'Technical detail',
        'source': 'Source Advisor report', 'analysis': 'Source analysis report',
        'no_goals': 'No numerical goal was supplied; no new target or hard constraint was created.',
        'no_metrics': 'No supported numerical measurement is available. Missing values were not treated as zero.',
        'separate': 'These measurements belong to separate records and protocols; they do not establish one jointly verified deployment condition.',
        'no_candidate': 'No optimization experiment is sufficiently supported yet.',
        'candidate': 'Conditional experiment candidate', 'review': 'Next evidence review',
        'not_authorized': 'Not authorized; no action will be started.',
        'not_executed': 'No workload or model change was executed during this reporting operation.',
        'authority': 'This is a presentation-only view; it does not replace the canonical AdvisorReport and is not authoritative evidence.',
        'original': 'Original technical record — free-form source text has not been translated.',
        'unknown_rule': 'Reliable localization is unavailable for this rule or version; consult the original technical record.',
        'unknown_metric': 'This finding has no supported typed metric description; no numerical meaning was guessed.',
        'partial': 'Some record associations are declared or missing; none were promoted to native verified associations.',
        'local_goal': 'A goal result describes only its source record; it is not deployment acceptance or a guarantee of future success.',
        'future': 'Future feasibility', 'support': 'Evidence support', 'input': 'Input validity', 'observed': 'Record-local outcome',
        'value': 'Recorded measurement', 'target': 'Target', 'consistency': 'Cross-goal consistency', 'hard': 'hard constraint', 'soft': 'preference',
        'no_actions': 'No additional review or experiment plan is present; no advice was fabricated to fill the gap.',
        'change': 'Variable to change', 'fixed': 'Fixed conditions', 'verify': 'How the outcome will be checked',
        'accept': 'Acceptance conditions', 'unknown_effect': 'Expected benefit and cost are unknown; no speedup or success percentage is supplied.',
        'experiment_fixed': 'Only batch size may change. Keep model, device, precision, backend, preprocessing and timing method fixed.',
        'experiment_verify': 'Check completed images/s, P95 latency, process RSS, linked quality evaluation and bounded stability at every selected batch.',
        'experiment_accept': 'Do not declare overall success until all explicit hard conditions pass on linked evidence. Independent confirmation and ComparisonContract are still required.',
        'experiment_missing': 'Explicit experiment budgets, finite batch points, execution capability and linked memory/quality acceptance remain prerequisites.',
        'experiment_risks': 'Larger batches can worsen latency, RAM use and OOM risk; quality and stability effects remain unknown.',
        'bindings': 'native: from producer evidence; declared: caller assertion; missing: unavailable. These labels are never promoted.',
        'technical_note': 'The complete record below retains identities, metric paths, scopes/protocols, eligibility, provenance, rule versions and acceptance boundaries. Source text is in its original language.',
        'counts': 'Finding counts', 'unknown_cost': 'This view does not grant a budget, cost estimate or execution permission.',
        'no_risk_translation': 'Additional source risks remain in the original technical record; translation does not create a new safety decision.',
    },
}

# Full sentence templates keyed by rule identity/version, not replacements in source text.
RULES = {
    'plan.latency-stage': {
        'tr': ('Gecikmeyi azaltmaya yönelik bir deney seçmeden önce mevcut P03/P04 işlem hattı ölçümlerini inceleyin.',
               'Modelin kendi gecikmesi ile uçtan uca süre farklı protokollerdir. Gecikmenin nedeni henüz kanıtlanmadı.',
               'Aşama sınırları, zamanlama kapsamı ve hedef kayda bağlanan kanıtlar eksik olabilir.'),
        'en': ('Inspect retained P03/P04 pipeline stages before choosing a latency experiment.',
               'Resident model latency and end-to-end time use different protocols. A causal bottleneck has not been established.',
               'Stage boundaries, clock scope and links to the goal capture may still be missing.'),
    },
    'plan.batch-throughput': {
        'tr': ('Batch optimizasyonunu seçmeden önce saklanan P06 ölçümlerini ve backend ilişkilerini doğrulayın.',
               'Tamamlanan görüntü hızı bir başlangıç ölçümüdür; batch büyümesi hızlanmayı garanti etmez.',
               'P06 kapsamı, açık backend/model ilişkileri ve deney koşulları doğrulanmalıdır.'),
        'en': ('Review retained P06 measurements and backend associations before selecting batch optimization.',
               'Completed images/s provides a baseline; a larger batch does not guarantee a speedup.',
               'P06 scope, explicit backend/model associations and experiment conditions need validation.'),
    },
    'plan.precision-quality': {
        'tr': ('Precision değişikliği önermeden önce mevcut P08 noktalarını ve kalite/kabiliyet kanıtlarını inceleyin.',
               'Her precision noktası ayrı kapsam taşır. BF16 otomatik hızlandırma değildir; desteklenmeyen INT8 kullanılabilir sayılmaz.',
               'P28 kalite ilişkileri, cohort ve ilgili hedef kayda ait native bağlar tamamlanmalıdır.'),
        'en': ('Review retained P08 points and quality/capability evidence before suggesting a precision change.',
               'Every precision point has a separate scope. BF16 is not automatically faster; unsupported INT8 is not runnable.',
               'P28 quality links, cohort associations and native links to the goal capture remain prerequisites.'),
    },
    'plan.memory-domain': {
        'tr': ('RAM kullanımını incelerken süreç RSS ölçümü ile modelin mantıksal boyutunu ayrı değerlendirin.',
               'Süreç RAM kullanımı model dışındaki maliyetleri de içerir. Model boyutu RAM veya VRAM değildir.',
               'Bellek mekanizması için ölçüm penceresi, allocation ömrü ve sahiplik kanıtları gerekebilir.'),
        'en': ('Review process RSS separately from the model logical footprint.',
               'Process RAM includes runtime costs. Model logical bytes are neither process RAM nor VRAM.',
               'Window, allocation-lifetime and ownership evidence may be needed to identify a memory mechanism.'),
    },
    'plan.accuracy-preservation': {
        'tr': ('Doğruluğun korunduğunu söylemeden önce baseline/candidate/cohort ilişkilerini doğrulayın.',
               'Mevcut test doğruluğu, model değiştikten sonra doğruluk kaybı olmayacağını kanıtlamaz.',
               'Bağlı model/veri/ön işleme kanıtları ve kullanıcının açık kalite kabul ölçütleri gerekir.'),
        'en': ('Validate baseline/candidate/cohort associations before claiming accuracy preservation.',
               'Recorded test accuracy does not prove there will be no quality loss after a change.',
               'Linked model/data/preprocessing evidence and explicit caller quality criteria are required.'),
    },
    'plan.hard-goal-consistency': {
        'tr': ('Deney seçmeden önce birbiriyle çelişen sert hedefleri düzeltin.',
               'Aynı kapsam için verilen hard koşullar birlikte karşılanamıyor; tekil kayıt sonuçları bunu değiştirmez.',
               'Kullanıcının açıkça belirlediği tutarlı bir sert hedef kümesi gerekir.'),
        'en': ('Resolve conflicting hard goals before selecting experiments.',
               'Hard conditions on the same scope cannot all hold; individual record outcomes do not resolve that conflict.',
               'A consistent explicit hard-goal set is required.'),
    },
}

BOUNDARIES = {
    'B4 native risk remains OPEN; no device/training authorization is granted.':
        ('B4 native çökme riski açık; cihaz veya eğitim çalışması için izin verilmedi.', 'B4 native crash risk remains OPEN; no device or training run is authorized.'),
    'qualified_stable_roof=not_established_by_this_report':
        ('P18 qualified roof kabulü bu raporla doğrulanmış değil.', 'P18 qualified stable-roof acceptance is not established by this report.'),
    'independently_verified_improvement=not_certified':
        ('P29 bağımsız doğrulanmış iyileşme kabulü yok.', 'P29 independently verified improvement is not certified.'),
    'long_run_7200_acceptance=not_certified':
        ('P31 7200 saniyelik dayanıklılık kabulü tamamlanmış sayılmaz.', 'P31 7200-second stability acceptance is not certified.'),
    'causal_memory_leak=not_proven':
        ('Nedensel bellek sızıntısı kanıtlanmış değil.', 'A causal memory leak has not been proven.'),
}

STATUS = {
    'valid': ('geçerli', 'valid'), 'invalid': ('geçersiz', 'invalid'), 'needs_scope': ('kapsam eksik', 'scope required'),
    'supported': ('bu yerel değerlendirme için kanıt var', 'supported for this local assessment'),
    'insufficient_evidence': ('kanıt yetersiz', 'insufficient evidence'),
    'unsupported_capability': ('bu kabiliyet desteklenmiyor', 'unsupported capability'),
    'observed_met': ('ilgili kayıt kapsamında karşılandı', 'met within the source record'),
    'observed_missed': ('ilgili kayıt kapsamında karşılanmadı', 'missed within the source record'),
    'unknown': ('bilinmiyor', 'unknown'), 'not_demonstrated': ('gösterilmedi', 'not demonstrated'),
    'demonstrated_in_scope': ('yalnız belirtilen kapsamda gösterildi', 'demonstrated only in the stated scope'),
    'no_conflict_identified': ('doğrudan çelişki bulunmadı', 'no direct conflict identified'),
    'potentially_conflicting': ('olası hedef çatışması', 'potentially conflicting'),
    'logically_inconsistent': ('mantıksal olarak çelişkili', 'logically inconsistent'),
    'not_evaluated': ('değerlendirilmedi', 'not evaluated'),
}

# DE/ES presentation strings; canonical evidence is intentionally never localized.
TEXT_DE_ES = {'de': {'title': 'HYSo — Modellleistungsbericht', 'performance': 'Aktuelle Modellleistung', 'goals': 'Status der Nutzerziele', 'findings': 'Wichtige Befunde', 'actions': 'Empfohlene nächste Prüfungen oder Experimente', 'why': 'Warum werden diese Schritte empfohlen?', 'missing': 'Fehlende Nachweise', 'risks': 'Risiken und Einschränkungen', 'execution': 'Ob eine Aktion ausgeführt wurde', 'technical': 'Technische Einzelheiten', 'source': 'Quellbericht des Advisors', 'analysis': 'Quellanalysebericht', 'no_goals': 'Es wurde kein numerisches Ziel angegeben; weder ein neues Ziel noch eine harte Bedingung wurde erstellt.', 'no_metrics': 'Keine unterstützte numerische Messung verfügbar. Fehlende Werte wurden nicht als null behandelt.', 'separate': 'Diese Messungen stammen aus getrennten Datensätzen und Protokollen; sie belegen keinen gemeinsam verifizierten Einsatzfall.', 'no_candidate': 'Die vorhandenen Nachweise reichen noch nicht aus, um ein Optimierungsexperiment zu empfehlen.', 'candidate': 'Bedingter Experimentkandidat', 'review': 'Nächste Nachweisprüfung', 'not_authorized': 'Nicht autorisiert; es wird keine Aktion gestartet.', 'not_executed': 'Während dieser Berichterstellung wurden weder Arbeitslasten ausgeführt noch Modelle verändert.', 'authority': 'Dies ist ausschließlich eine Darstellung; sie ersetzt den kanonischen AdvisorReport nicht und gilt nicht als maßgeblicher Nachweis.', 'original': 'Ursprünglicher technischer Datensatz — Freitext wurde nicht übersetzt.', 'unknown_rule': 'Für diese Regel oder Version liegt keine zuverlässige Übersetzung vor; prüfen Sie den ursprünglichen technischen Datensatz.', 'unknown_metric': 'Für diesen Befund liegt keine unterstützte typisierte Metrikbeschreibung vor; die numerische Bedeutung wurde nicht geraten.', 'partial': 'Einige Datensatzverknüpfungen sind deklariert oder fehlen; keine wurde als nativ verifiziert eingestuft.', 'local_goal': 'Ein Zielergebnis gilt nur für seinen Quelldatensatz; es ist weder eine Einsatzfreigabe noch eine Garantie künftigen Erfolgs.', 'future': 'Zukünftige Machbarkeit', 'support': 'Nachweisunterstützung', 'input': 'Gültigkeit der Eingabe', 'observed': 'Ergebnis im Datensatz', 'value': 'Erfasster Messwert', 'target': 'Zielwert', 'consistency': 'Konsistenz der Ziele', 'hard': 'harte Bedingung', 'soft': 'Präferenz', 'no_actions': 'Es liegt kein zusätzlicher Prüf- oder Experimentplan vor; es wurden keine Empfehlungen erfunden.', 'change': 'Zu ändernde Variable', 'fixed': 'Unveränderliche Bedingungen', 'verify': 'Wie wird das Ergebnis geprüft?', 'accept': 'Abnahmekriterien', 'unknown_effect': 'Erwarteter Nutzen und Kosten sind unbekannt; weder Beschleunigung noch Erfolgsquote werden angegeben.', 'experiment_fixed': 'Nur die Batchgröße darf verändert werden. Modell, Gerät, Präzision, Backend, Vorverarbeitung und Messmethode bleiben konstant.', 'experiment_verify': 'Für jede ausgewählte Batchgröße müssen Bilder/s, P95-Latenz, Prozess-RSS, verknüpfte Qualitätsbewertung und begrenzte Stabilität erneut geprüft werden.', 'experiment_accept': 'Keinen Gesamterfolg erklären, bevor alle expliziten harten Bedingungen mit verknüpften Nachweisen erfüllt sind. Unabhängige Bestätigung und ComparisonContract bleiben erforderlich.', 'experiment_missing': 'Explizite Experimentbudgets, begrenzte Batchwerte, Ausführungskapazität sowie verknüpfte Speicher- und Qualitätsnachweise sind erforderlich.', 'experiment_risks': 'Größere Batches können Latenz, RAM-Bedarf und OOM-Risiko erhöhen; Auswirkungen auf Qualität und Stabilität bleiben unbekannt.', 'bindings': 'native: aus Erzeugernachweis; declared: Nutzeraussage; missing: nicht verfügbar. Diese Kategorien werden niemals hochgestuft.', 'technical_note': 'Der vollständige Datensatz unten bewahrt Identitäten, Metrikpfade, Geltungsbereiche/Protokolle, Eignung, Herkunft, Regelversionen und Abnahmegrenzen. Quelltexte bleiben in der Originalsprache.', 'counts': 'Anzahl der Befunde', 'unknown_cost': 'Diese Ansicht erteilt weder Budget noch Kostenschätzung oder Ausführungserlaubnis.', 'no_risk_translation': 'Zusätzliche Quellrisiken bleiben im ursprünglichen technischen Datensatz; eine Übersetzung erzeugt keine neue Sicherheitsentscheidung.'}, 'es': {'title': 'HYSo — Informe de rendimiento del modelo', 'performance': 'Rendimiento actual del modelo', 'goals': 'Estado de los objetivos del usuario', 'findings': 'Hallazgos importantes', 'actions': 'Próximas revisiones o experimentos sugeridos', 'why': '¿Por qué se sugieren estos pasos?', 'missing': 'Evidencia faltante', 'risks': 'Riesgos y limitaciones', 'execution': 'Si se realizó alguna acción', 'technical': 'Detalles técnicos', 'source': 'Informe Advisor de origen', 'analysis': 'Informe de análisis de origen', 'no_goals': 'No se proporcionó ningún objetivo numérico; no se creó ningún objetivo nuevo ni restricción estricta.', 'no_metrics': 'No hay mediciones numéricas compatibles disponibles. Los valores faltantes no se trataron como cero.', 'separate': 'Estas mediciones pertenecen a registros y protocolos distintos; no establecen una condición de despliegue verificada conjuntamente.', 'no_candidate': 'La evidencia disponible aún no es suficiente para recomendar un experimento de optimización.', 'candidate': 'Experimento candidato condicionado', 'review': 'Próxima revisión de evidencia', 'not_authorized': 'No autorizado; no se iniciará ninguna acción.', 'not_executed': 'Durante la generación de este informe no se ejecutó ninguna carga de trabajo ni se modificó ningún modelo.', 'authority': 'Esta vista es solo de presentación; no sustituye al AdvisorReport canónico ni constituye evidencia autorizada.', 'original': 'Registro técnico original — no se tradujo el texto libre de origen.', 'unknown_rule': 'No existe una traducción fiable para esta regla o versión; consulte el registro técnico original.', 'unknown_metric': 'Este hallazgo no tiene una descripción de métrica tipada compatible; no se supuso su significado numérico.', 'partial': 'Algunas asociaciones de registros son declaradas o faltantes; ninguna se elevó a verificada de forma nativa.', 'local_goal': 'El resultado de un objetivo solo describe su registro de origen; no supone aceptación del despliegue ni garantiza el éxito futuro.', 'future': 'Viabilidad futura', 'support': 'Respaldo de la evidencia', 'input': 'Validez de la entrada', 'observed': 'Resultado dentro del registro', 'value': 'Medición registrada', 'target': 'Objetivo', 'consistency': 'Coherencia entre objetivos', 'hard': 'restricción estricta', 'soft': 'preferencia', 'no_actions': 'No existe un plan adicional de revisión o experimentación; no se inventaron recomendaciones para cubrir la ausencia.', 'change': 'Variable que se modificará', 'fixed': 'Condiciones fijas', 'verify': '¿Cómo se comprobará el resultado?', 'accept': 'Criterios de aceptación', 'unknown_effect': 'Se desconocen los beneficios y costos esperados; no se indica aceleración ni porcentaje de éxito.', 'experiment_fixed': 'Solo puede cambiar el tamaño del lote. Mantenga constantes el modelo, dispositivo, precisión, backend, preprocesamiento y método de medición.', 'experiment_verify': 'Para cada tamaño de lote seleccionado, vuelva a comprobar imágenes/s, latencia P95, RSS del proceso, evaluación de calidad vinculada y estabilidad acotada.', 'experiment_accept': 'No declare éxito global hasta que se cumplan todas las condiciones estrictas explícitas con evidencia vinculada. Siguen siendo necesarios la confirmación independiente y ComparisonContract.', 'experiment_missing': 'Son requisitos previos los presupuestos explícitos, puntos de lote finitos, capacidad de ejecución y evidencia vinculada de aceptación de memoria y calidad.', 'experiment_risks': 'Los lotes mayores pueden aumentar latencia, uso de RAM y riesgo de falta de memoria; se desconocen los efectos sobre calidad y estabilidad.', 'bindings': 'native: procede de la evidencia del productor; declared: afirmación del usuario; missing: no disponible. Nunca se promueven estas categorías.', 'technical_note': 'El registro completo conserva identidades, rutas de métricas, ámbitos/protocolos, elegibilidad, procedencia, versiones de reglas y límites de aceptación. El texto de origen permanece en su idioma original.', 'counts': 'Cantidad de hallazgos', 'unknown_cost': 'Esta vista no concede presupuesto, estimación de costos ni permiso de ejecución.', 'no_risk_translation': 'Los riesgos adicionales permanecen en el registro técnico original; traducirlos no genera una nueva decisión de seguridad.'}}
RULES_DE_ES = {'plan.latency-stage': {'de': ('Prüfen Sie vorhandene P03/P04-Pipelinephasen, bevor Sie ein Latenzexperiment auswählen.', 'Modelllatenz und End-to-End-Zeit folgen verschiedenen Protokollen. Ein kausaler Engpass ist nicht belegt.', 'Phasengrenzen, Zeitmessungsumfang und Zielnachweise können noch fehlen.'), 'es': ('Revise las etapas P03/P04 registradas antes de elegir un experimento de latencia.', 'La latencia del modelo y el tiempo de extremo a extremo usan protocolos diferentes. No se ha demostrado un cuello de botella causal.', 'Pueden faltar límites de etapas, ámbito de medición temporal y vínculos con el objetivo.')}, 'plan.batch-throughput': {'de': ('Prüfen Sie vorhandene P06-Messwerte und Backend-Verknüpfungen vor einer Batchoptimierung.', 'Die gemessene Bildrate ist ein Ausgangswert; größere Batches garantieren keine Beschleunigung.', 'P06-Geltungsbereich, Backend-/Modellverknüpfungen und Experimentbedingungen sind zu prüfen.'), 'es': ('Revise las mediciones P06 y asociaciones del backend antes de optimizar el lote.', 'La tasa de imágenes completadas es una referencia; un lote mayor no garantiza acelerar el proceso.', 'Deben validarse el alcance P06, las asociaciones backend/modelo y las condiciones del experimento.')}, 'plan.precision-quality': {'de': ('Prüfen Sie P08-Präzisionspunkte und Qualitäts-/Fähigkeitsnachweise vor einem Präzisionswechsel.', 'Jeder Präzisionspunkt hat einen eigenen Geltungsbereich. BF16 ist nicht automatisch schneller; nicht unterstütztes INT8 ist nicht ausführbar.', 'P28-Qualitätsverknüpfungen, Kohorten und native Zielnachweise sind erforderlich.'), 'es': ('Revise los puntos P08 y la evidencia de calidad/capacidad antes de cambiar la precisión.', 'Cada punto de precisión tiene su propio alcance. BF16 no es automáticamente más rápido; INT8 no compatible no puede ejecutarse.', 'Se requieren vínculos P28 de calidad, cohortes y evidencia nativa del objetivo.')}, 'plan.memory-domain': {'de': ('Betrachten Sie Prozess-RSS und die logische Modellgröße getrennt.', 'Prozess-RAM enthält Laufzeitkosten. Die logische Modellgröße ist weder Prozess-RAM noch VRAM.', 'Messfenster, Lebensdauer von Allokationen und Eigentumsnachweise können nötig sein.'), 'es': ('Examine el RSS del proceso por separado del tamaño lógico del modelo.', 'La RAM del proceso incluye costos de ejecución. El tamaño lógico del modelo no equivale a RAM del proceso ni a VRAM.', 'Pueden requerirse ventana de medición, vida de asignaciones y evidencia de pertenencia.')}, 'plan.accuracy-preservation': {'de': ('Prüfen Sie Baseline-/Kandidaten-/Kohortenverknüpfungen, bevor Sie erhaltene Genauigkeit behaupten.', 'Erfasste Testgenauigkeit beweist nicht, dass ein Modellwechsel keinen Qualitätsverlust bewirkt.', 'Verknüpfte Modell-/Daten-/Vorverarbeitungsnachweise und explizite Qualitätskriterien sind erforderlich.'), 'es': ('Valide los vínculos entre referencia, candidato y cohorte antes de afirmar que se conserva la exactitud.', 'La exactitud registrada no demuestra que un cambio de modelo evite pérdidas de calidad.', 'Se requieren evidencias vinculadas de modelo/datos/preprocesamiento y criterios explícitos de calidad.')}, 'plan.hard-goal-consistency': {'de': ('Beheben Sie widersprüchliche harte Ziele, bevor Sie Experimente auswählen.', 'Harte Bedingungen im selben Geltungsbereich lassen sich nicht gleichzeitig erfüllen; einzelne Datensätze lösen den Widerspruch nicht.', 'Ein konsistenter Satz expliziter harter Ziele ist erforderlich.'), 'es': ('Resuelva los objetivos estrictos contradictorios antes de elegir experimentos.', 'Las condiciones estrictas del mismo ámbito no pueden cumplirse simultáneamente; los resultados individuales no resuelven el conflicto.', 'Se necesita un conjunto coherente y explícito de objetivos estrictos.')}}
BOUNDARIES_DE_ES = {'B4 native risk remains OPEN; no device/training authorization is granted.': ('Das native B4-Risiko bleibt OFFEN; Geräte- oder Trainingsausführung ist nicht autorisiert.', 'El riesgo nativo B4 sigue ABIERTO; no se autoriza ejecutar dispositivos ni entrenamiento.'), 'qualified_stable_roof=not_established_by_this_report': ('Die qualifizierte P18-Stable-Roof-Abnahme ist durch diesen Bericht nicht belegt.', 'Este informe no demuestra la aceptación de techo estable cualificado P18.'), 'independently_verified_improvement=not_certified': ('Die unabhängig verifizierte Verbesserung P29 ist nicht zertifiziert.', 'No está certificada la mejora P29 verificada de forma independiente.'), 'long_run_7200_acceptance=not_certified': ('Die 7200-Sekunden-Stabilitätsabnahme P31 ist nicht zertifiziert.', 'No está certificada la aceptación de estabilidad P31 de 7200 segundos.'), 'causal_memory_leak=not_proven': ('Ein kausales Speicherleck ist nicht bewiesen.', 'No se ha demostrado una fuga de memoria causal.')}
STATUS_DE_ES = {'valid': ('gültig', 'válido'), 'invalid': ('ungültig', 'inválido'), 'needs_scope': ('Geltungsbereich erforderlich', 'se requiere ámbito'), 'supported': ('für diese lokale Bewertung belegt', 'respaldado para esta evaluación local'), 'insufficient_evidence': ('unzureichende Nachweise', 'evidencia insuficiente'), 'unsupported_capability': ('Fähigkeit nicht unterstützt', 'capacidad no compatible'), 'observed_met': ('im Quelldatensatz erfüllt', 'cumplido en el registro de origen'), 'observed_missed': ('im Quelldatensatz nicht erfüllt', 'no cumplido en el registro de origen'), 'unknown': ('unbekannt', 'desconocido'), 'not_demonstrated': ('nicht nachgewiesen', 'no demostrado'), 'demonstrated_in_scope': ('nur im angegebenen Geltungsbereich nachgewiesen', 'demostrado solo en el ámbito indicado'), 'no_conflict_identified': ('kein direkter Widerspruch erkannt', 'no se identificó conflicto directo'), 'potentially_conflicting': ('möglicher Zielkonflikt', 'posible conflicto de objetivos'), 'logically_inconsistent': ('logisch widersprüchlich', 'lógicamente incoherente'), 'not_evaluated': ('nicht bewertet', 'no evaluado')}

for _lang in ('de', 'es'):
    if set(TEXT_DE_ES[_lang]) != set(TEXT['en']):
        raise RuntimeError('advisor_locale_keys_incomplete')
    TEXT[_lang] = TEXT_DE_ES[_lang]
for _rule, _localized in RULES_DE_ES.items():
    RULES[_rule].update(_localized)
# Leave existing bilingual status and boundary tuples unchanged; optional extra translations follow.
