"""Idempotent E01-E06 localized guide registration, no runtime imports."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def update():
    path=ROOT/'docs/user-guide/content.json';data=json.loads(path.read_text(encoding='utf-8'))
    topic=dict(id='export.overview',related=['analysis.p35','api.python-cli','support.cpu-gpu'],
        sources=['hyso/export/__init__.py','hyso/export/contracts.py','hyso/export/converters.py','hyso/export/native.py','docs/guides/model-export.md'],
        inventory=['hyso.export','E01','E02','E03','E04','E05','E06'],translations={
        'tr':dict(title='Model export ve deployment: E01–E06',what='hyso.export, açık kullanıcı isteğiyle kaynak modeli koruyarak ayrı ONNX, TorchScript, ExportedProgram veya kontrollü TensorRT artifact üretir. Analyzer ve Advisor model değiştirmez.',
            when='Modeli başka bir runtime üzerinde kullanmak istediğinizde onnx, tensorrt, pytorch_tensorrt, libtorch veya torch_export çağrısını açıkça seçin.',
            benefit='Hash manifesti, inference kopyası, gerçek çıktı eşdeğerliği ve dokuz ayrı aşama sonucu kaydedilir. Varsayılan overwrite kapalıdır; çok dosyalı çıktılar commit manifestiyle okunmalıdır.',
            limits='TensorRT 10.x, gerçek UUID/DLL admission ve owned worker gerektirir; bu ortamda bağımlılık yoktur. Python TorchScript C++ kanıtı değildir. Windows Torch 2.5 AOTInductor kapalıdır. Trace yalnız örnek dalları kapsar. E07–E13 ve tam CLI/Advisor entegrasyonu sonraki kapsamdır. Teknik hata metinleri İngilizce; code/message_key sabittir. full_cuda_acceptance=false.'),
        'en':dict(title='Model export and deployment: E01–E06',what='hyso.export explicitly creates separate ONNX, TorchScript, ExportedProgram or controlled TensorRT artifacts while preserving the source model. Analyzer and Advisor do not modify models.',
            when='Explicitly choose onnx, tensorrt, pytorch_tensorrt, libtorch or torch_export when deploying a model to another runtime.',
            benefit='Hash manifests, inference copies, real output equivalence and nine separate stage outcomes are retained. Overwrite is off by default; multi-file artifacts require their commit manifest.',
            limits='TensorRT 10.x requires real UUID/DLL admission and an owned worker; its dependency is absent here. Python TorchScript is not C++ proof. Windows Torch 2.5 AOTInductor is closed. Trace covers example branches only. E07–E13 and full CLI/Advisor integration remain future scope. Technical errors are English; code/message_key are stable. full_cuda_acceptance=false.'),
        'de':dict(title='Modellexport und Deployment: E01–E06',what='hyso.export erstellt auf ausdrücklichen Wunsch separate ONNX-, TorchScript-, ExportedProgram- oder kontrollierte TensorRT-Artefakte und bewahrt das Quellmodell. Analyzer und Advisor ändern keine Modelle.',
            when='Wählen Sie onnx, tensorrt, pytorch_tensorrt, libtorch oder torch_export ausdrücklich für die Ausführung in einer anderen Laufzeit.',
            benefit='Hash-Manifeste, Inferenzkopien, echte Ausgabeäquivalenz und neun getrennte Stufenergebnisse bleiben erhalten. Überschreiben ist standardmäßig aus; mehrteilige Artefakte benötigen ihr Commit-Manifest.',
            limits='TensorRT 10.x benötigt echte UUID-/DLL-Zulassung und einen eigenen Worker; die Abhängigkeit fehlt hier. Python TorchScript beweist keine C++-Ausführung. Windows Torch 2.5 AOTInductor bleibt geschlossen. Trace erfasst nur Beispielzweige. E07–E13 und vollständige CLI-/Advisor-Integration folgen später. Technische Fehler sind Englisch; code/message_key bleiben stabil. full_cuda_acceptance=false.'),
        'es':dict(title='Exportación y despliegue: E01–E06',what='hyso.export crea explícitamente artefactos ONNX, TorchScript, ExportedProgram o TensorRT controlados por separado y conserva el modelo original. Analyzer y Advisor no modifican modelos.',
            when='Seleccione explícitamente onnx, tensorrt, pytorch_tensorrt, libtorch o torch_export para usar otro runtime.',
            benefit='Se conservan manifiestos hash, copias de inferencia, equivalencia real de salidas y nueve resultados de etapa separados. La sobrescritura está desactivada; los artefactos múltiples requieren su manifiesto de commit.',
            limits='TensorRT 10.x requiere admisión real UUID/DLL y worker propio; la dependencia no existe aquí. Python TorchScript no prueba C++. Windows Torch 2.5 AOTInductor sigue cerrado. Trace solo cubre ramas de ejemplo. E07–E13 y la integración completa CLI/Advisor quedan pendientes. Los errores técnicos están en inglés; code/message_key son estables. full_cuda_acceptance=false.')})
    topic['inventory'] += ['E07','E08','E09','E10','E11','E12']
    updates={
      'en':('E07–E12: ONNX mixed FP16, dynamic/static INT8 QDQ, cohort numerical/labelled quality validation, deployment benchmarks, compatible comparisons, Advisor, CLI and JSON/Markdown/HTML reports.',
            'TensorRT execution requires a separately admitted interpreter, RECORD-backed DLL identities and owned workers. Native GPU development remains NOT_TESTED. CPU FP16 cast coverage may be BLOCKED. Unmeasured metrics are unavailable. full_cuda_acceptance=false.'),
      'tr':('E07–E12: ONNX mixed FP16, dynamic/static INT8 QDQ, cohort sayısal/etiketli kalite doğrulaması, deployment benchmark, uyumlu karşılaştırma, Advisor, CLI ve JSON/Markdown/HTML raporları.',
            'TensorRT ayrı kabul edilmiş interpreter, RECORD tabanlı DLL kimlikleri ve owned worker gerektirir. Native GPU geliştirmesi NOT_TESTED. CPU FP16 cast eşlemesi BLOCKED olabilir. Ölçülmeyen metrikler unavailable. full_cuda_acceptance=false.'),
      'de':('E07–E12: ONNX Mixed-FP16, dynamisches/statisches INT8 QDQ, numerische Kohorten- und Aufgabenqualität, Deployment-Benchmarks, kompatible Vergleiche, Advisor, CLI und Berichte.',
            'TensorRT benötigt einen separat zugelassenen Interpreter, RECORD-geprüfte DLLs und eigene Worker. Native GPU-Entwicklung: NOT_TESTED. FP16-CPU-Zuordnung kann BLOCKED sein. Nicht gemessene Metriken: unavailable. full_cuda_acceptance=false.'),
      'es':('E07–E12: ONNX FP16 mixto, INT8 dinámico/estático QDQ, validación numérica y calidad con etiquetas, benchmarks, comparaciones compatibles, Advisor, CLI e informes.',
            'TensorRT requiere un intérprete admitido por separado, DLL verificadas mediante RECORD y workers propios. Desarrollo GPU nativo: NOT_TESTED. La cobertura FP16 CPU puede quedar BLOCKED. Métricas no medidas: unavailable. full_cuda_acceptance=false.')}
    for lang,(benefit,limits) in updates.items():
        topic['translations'][lang]['benefit'] += ' '+benefit
        topic['translations'][lang]['limits']=limits
    data['topics']=[t for t in data['topics'] if t['id']!='export.overview']+[topic]
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'hyso/data/api-topics.json';index=json.loads(p.read_text(encoding='utf-8'))
    for name in ('hyso.export','hyso.export.ExportConfig','hyso.export.NativeConfig','hyso.export.ExportResult','hyso.export.ExportError',
        'hyso.export.availability','hyso.export.onnx','hyso.export.tensorrt','hyso.export.pytorch_tensorrt','hyso.export.libtorch','hyso.export.torch_export','hyso.export.aotinductor',
        'hyso.export.fp16','hyso.export.quantize','hyso.export.Runtime','hyso.export.Tolerance','hyso.export.output_metrics',
        'hyso.export.classification_metrics','hyso.export.validate','hyso.export.benchmark','hyso.export.compare','hyso.export.advise','hyso.export.render','hyso.export.save','hyso.export.load',
        'hyso.advisor.export_advice','hyso.reporting.render_export'):
        index['entries'][name]=dict(topic='export.overview',canonical=name,kind='export_api',origin='E01_E06_explicit_public_namespace')
    p.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':update()
