# M2-P-B4 — Conditional acceptance (2026-10-20)

**B4: CONDITIONAL GO — development continuation only**

Kullanıcı, kök nedeni bilinmeyen native failure riskini açıkça kabul ederek
B5/P17 geliştirmesine devam edilmesini onayladı. Bu kayıt yeni özellik kodundan
önce oluşturuldu; önceki B4/R1/R2/R3 audit belgeleri değiştirilmedi.

Korunmuş tarihsel kanıtlar: P06/P07/P08 desteklenen kapsamda PASS; full CPU
2313 PASS / 1 expected SKIP; R1 clean native gate 196 distinct PASS / 198
successful executions. Bunlar B5 kaynakları üzerinde yeni çalıştırılmış testler
değildir. Orijinal 0xC0000409, reason 2 STACK_COOKIE_CHECK_FAILURE olayı geçerli
tarihsel başarısızlıktır. R3 stack'i SiLU forward → PyTorch CUDA dispatch →
NVIDIA driver yolu ile uyumludur; HYSo, PyTorch veya NVIDIA kaynaklı kök neden
kanıtlanmadı. **Root cause: UNRESOLVED.** Production düzeltmesi uygulanmadı.

Bu koşullu karar orijinal strict gate'in PASS olduğu, crash'in giderildiği veya
PyPI/public stable release onayı anlamına gelmez. Yalnız açık kullanıcı risk
kabulüne dayanan development continuation kararıdır.

**Reliability risk: OPEN.** Tekrarlayan native crash, güvenilir biçimde üretilen
regresyon veya model/state corruption saptanırsa B5 kalite kapısı ve geliştirme
durdurulur; NO-GO raporlanır. Kararlı yayın öncesinde bu açık risk için ayrı
release-gate değerlendirmesi zorunludur. B5 başarısı bu riski kapatmaz.

İlgili korunmuş belgeler: [B4](m2-p-b4-sweeps-precision-final-gate.md),
[R1](m2-p-b4-r1-native-replay-recovery.md),
[R2](m2-p-b4-r2-native-failfast-diagnostics.md),
[R3](m2-p-b4-r3-native-callpath-investigation.md).
