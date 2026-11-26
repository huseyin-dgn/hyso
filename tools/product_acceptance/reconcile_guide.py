"""Idempotent localized scope note reconciliation; no runtime or campaign runs."""
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
NOTES={
    'tr':'Güncel sınırlı CUDA kabulü: kaynak ve artifact hash bağlı kanıtlar support registry current_cuda_acceptance alanındadır. Yalnız kayıtlı model/cihaz/runtime/protokolü kapsar; genel donanım sertifikası değildir. P18 ileri sayaçlar, P23 saved activation/B4, P31 CUDA 7200s ve P33 ileri enerji kapsamı kapalıdır; full_cuda_acceptance=false.',
    'en':'Current bounded CUDA acceptance: source/artifact-hash-bound evidence is in support registry current_cuda_acceptance. It covers only the recorded model/device/runtime/protocol, not general hardware certification. P18 advanced counters, P23 saved activation/B4, P31 CUDA 7200s and P33 advanced energy remain closed; full_cuda_acceptance=false.',
    'de':'Aktuelle begrenzte CUDA-Abnahme: Nachweise mit Quell- und Artefakt-Hashes stehen im Supportregister unter current_cuda_acceptance. Sie gelten nur für das aufgezeichnete Modell/Gerät/Laufzeit/Protokoll, ohne allgemeine Hardwarezertifizierung. P18 erweiterte Zähler, P23 saved activation/B4, P31 CUDA 7200s und P33 erweiterte Energie bleiben geschlossen; full_cuda_acceptance=false.',
    'es':'Aceptación CUDA acotada actual: las evidencias vinculadas a hashes de fuentes y artefactos están en current_cuda_acceptance del registro de soporte. Solo cubren el modelo/dispositivo/runtime/protocolo registrado, sin certificación general de hardware. P18 contadores avanzados, P23 saved activation/B4, P31 CUDA 7200s y P33 energía avanzada siguen cerrados; full_cuda_acceptance=false.'}
LIMITS={
    'tr':'Advisor eğitim/capture başlatmaz, model kodunu değiştirmez ve kazanç garantisi vermez. Öneriler ayrıca ölçülmelidir. Rehber, rapor, Advisor ve karşılaştırma sunumları tr/en/de/es destekler. Python docstring ve argparse help çoğunlukla İngilizcedir.',
    'en':'Advisor does not launch training/capture, edit model code or guarantee gains. Suggestions require separate measurements. Guide, report, Advisor and comparison presentation support tr/en/de/es. Python docstrings and argparse help are primarily English.',
    'de':'Advisor startet weder Training noch Capture, verändert keinen Modellcode und garantiert keinen Gewinn. Vorschläge erfordern separate Messungen. Handbuch, Berichte, Advisor und Vergleichsdarstellung unterstützen tr/en/de/es. Python-Docstrings und argparse-Hilfe sind überwiegend Englisch.',
    'es':'Advisor no inicia entrenamientos ni capturas, no edita código ni garantiza mejoras. Las propuestas requieren mediciones separadas. Guía, informes, Advisor y presentación de comparaciones admiten tr/en/de/es. Los docstrings de Python y la ayuda argparse están principalmente en inglés.'}

def reconcile():
    path=REPO/'docs/user-guide/content.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    # Remove only this note, including its earlier legacy-shell encoding form.
    for topic in data['topics']:
        for lang,fields in topic['translations'].items():
            if topic['id']=='advisor.recommendations':fields['limits']=LIMITS[lang]
            if topic['id']=='support.cpu-gpu' or topic['id'].startswith('analysis.p'):
                prefix=NOTES[lang].split(':',1)[0]
                text=fields['limits']
                for marker in (prefix,prefix.encode('ascii','replace').decode('ascii')):
                    text=text.split(' '+marker,1)[0]
                fields['limits']=text+' '+NOTES[lang]
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':reconcile()
