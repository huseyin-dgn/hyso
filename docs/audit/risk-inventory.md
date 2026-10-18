# HYSo P0.1-A — Kritik teknik risk envanteri

Referans `b74e3e2cdec527e7bb7b078f69636a27b19923fe`, main, 18 Ekim 2026. Hiçbir bulgu düzeltilmedi; regression testi yazılmadı. P0.1-C test önerileri gelecekte yapılacak işlerdir. P1 yüksek, P2 orta önceliktir. Statik kaynak olgusu ile çalıştırılmış kullanıcı senaryosu birbirinden ayrılmıştır.

## Runtime doğrulanmış hata ve gözlemler

### R01 — EBO doğrudan çağrılamıyor (P1)

- **Dosya/satır:** `hyso/models/bcam.py:184-205,270-539; ELO forward 171-178`.
- **Teknik açıklama:** EBO nn.Module wrapper ama forward tanımı yok; fit child modeli kullanır.
- **Kullanıcı etkisi:** ebo(x), dış model araçları ve wrapper export akışları hata verir; ebo.model(x) ayrı yüzeydir.
- **Doğrulama durumu:** İki küçük CPU EBO konfigürasyonunda NotImplementedError; child aynı girdide [2,2] çıktı verdi. Constructor/import başarılı.
- **P0.1-C test önerisi:** Wrapper ve child çağrılarını ayrı sözleşmeler olarak sabitle; ileride forward eklenirse hooks ve state prefix değişmezliğini de denetle.

### R02 — EarlyStopping best_state canlı tensor storage kullanıyor (P1)

- **Dosya/satır:** `hyso/callbacks/callbacks.py:14-18,26-28`.
- **Teknik açıklama:** best_state=model.state_dict() derin kopya almaz; sonraki in-place güncellemeler saklanan ağırlığı değiştirir.
- **Kullanıcı etkisi:** Restore en iyi epoch ağırlığına dönemeyebilir; kullanıcı yanlış modeli seçebilir.
- **Doğrulama durumu:** CPU Linear ile ilk callback sonrası parametreye +1 eklendi; best_state data_ptr parametreyle aynı, saklanan değer değişti. Optimizer/fit/restore çalıştırılmadı.
- **P0.1-C test önerisi:** En iyi snapshot sonrası kötüleşen optimizer adımları ve restore; parametre ve buffer değerleri ile storage bağımsızlığını kontrol et.

### R03 — EMA buffers güncellenmiyor ve gölge parent state dışında (P1)

- **Dosya/satır:** `hyso/models/WithSE.py:20-43,84-99,149-152,183-187`.
- **Teknik açıklama:** update yalnızca zip(parameters) üzerinde çalışır. EMA nn.Module değildir; gölge parent traversal/state_dict dışında kalır.
- **Kullanıcı etkisi:** BatchNorm running_mean/var/count eski kalabilir; validation ayrışabilir. Parent save/to/eval gölge durumunu otomatik yönetmez.
- **Doğrulama durumu:** CPU CEMO canlı running_mean +3 sonrası update gölge buffer değerini değiştirmedi, gölge parametre değişti. EMA açık parent 58 girdili, ema.* yok; gerçek eğitim başarımı ölçülmedi.
- **P0.1-C test önerisi:** BN floating ve integer buffer politikası, device taşıma ve save/restore; zip isim/shape eşlemesi ve sessiz kırpılma sınırlarını denetle.

### R04 — EBO wrapper ve child checkpoint prefix farkı (P1)

- **Dosya/satır:** `hyso/models/bcam.py:190-199,254,524; hyso/callbacks/callbacks.py:40`.
- **Teknik açıklama:** Wrapper state model.* iken model-level callback child ELO state kaydeder.
- **Kullanıcı etkisi:** Kaydetme/yükleme hedefi karıştırılırsa missing/unexpected keys hatası; fit checkpoint ile wrapper checkpoint aynı şema değil.
- **Doğrulama durumu:** İki küçük EBO wrapper state child strict load denemesi RuntimeError verdi; eş config wrapper→wrapper strict yükleme başarılı. Dosya save/load veya tarihsel dosya denenmedi.
- **P0.1-C test önerisi:** Wrapper/child/callback payload roundtrip ayrı; yanlış hedefin strict reddi ve olası prefix göç kararını açık test et.

### R05 — Base ortamda paket importu için torch yok (P1 ortam)

- **Dosya/satır:** `hyso/__init__.py:1; hyso/models/bcam.py:3`.
- **Teknik açıklama:** Root eager import models yükler; etkin Python 3.13.12 ortamı torch içermiyor.
- **Kullanıcı etkisi:** Etkin Python ile hedef API import edilemiyor; bu tek başına üretim kodu hatası değildir.
- **Doğrulama durumu:** Base import hyso ModuleNotFoundError; hazır torch_gpu Python 3.10.18 ile normal hyso ve hedef on sembol importu başarılı. Stub/bypass kullanılmadı.
- **P0.1-C test önerisi:** Temiz dependency kurulumu ve interpreter matrisi; eksik bağımlılık hata mesajı ve canonical import smoke kontrolü.

## Statik analiz bulguları — kullanıcı senaryosu çalıştırılmadı

### R06 — Validation rastgele augmentation içeriyor (P1)

- **Dosya/satır:** `hyso/augmentation/augmentation.py:32-38,90-96`.
- **Teknik açıklama:** Ayrı val_path için val_transform basic kullanır; basic RandomHorizontalFlip/RandomRotation/ColorJitter içerir.
- **Kullanıcı etkisi:** Validation metriği, early stopping ve model seçimi girdiden dolayı değişebilir.
- **Doğrulama durumu:** Statik kod yolu; ek olarak basic Compose türleri runtime oluşturulup kaydedildi. Aynı görüntünün tekrarlı dönüşümü ölçülmedi.
- **P0.1-C test önerisi:** Ayrı val_path ile val transform sözleşmesi ve aynı örneğin tekrarlı determinismi; seed etkisini ayrı ölç.

### R07 — Random split ortak train transform ve kontrolsüz RNG kullanıyor (P1)

- **Dosya/satır:** `hyso/augmentation/augmentation.py:94-105`.
- **Teknik açıklama:** random_split aynı ImageFolder iki Subset oluşturur; train_transform paylaşılır, hazırlanan val_transform kullanılmaz. Generator/seed/stratification API yok; val_size=floor(0.2*N).
- **Kullanıcı etkisi:** Advanced/extra_ops validation tarafına taşınabilir; split dış RNG durumuna bağlıdır. N<5 için val boş, sınıf dengesizliği olabilir. İndeks çakışması veya veri sızıntısı kanıtı değildir.
- **Doğrulama durumu:** Statik; dataset/indeksler çalıştırılmadı.
- **P0.1-C test önerisi:** İndeks ayrıklığı/tamlığı, transform paylaşımı, seed reproducibility, küçük dataset ve azınlık sınıfı senaryoları.

### R08 — Ayrı ImageFolder sınıf haritaları karşılaştırılmıyor (P1)

- **Dosya/satır:** `hyso/augmentation/augmentation.py:94-96,107`.
- **Teknik açıklama:** Train ve val bağımsız class_to_idx oluşturur; eşitliği kontrol edilmez, sınıf haritası ayrı return edilmez.
- **Kullanıcı etkisi:** Bir splitte farklı/eksik sınıf klasörü hedef indekslerinin farklı yorumlanmasına yol açabilir.
- **Doğrulama durumu:** Statik kontrol eksikliği; gerçek veri senaryosu denenmedi.
- **P0.1-C test önerisi:** Eksik/farklı sınıf klasörlü küçük sentetik dataset ile açık eşleme veya red davranışı.

### R09 — Gradient accumulation son kısmi pencereyi kaybediyor (P1)

- **Dosya/satır:** `hyso/models/bcam.py:357-389,411`.
- **Teknik açıklama:** Optimizer step yalnızca (step+1)%accum_steps==0; epoch sonu flush yok. Sonraki epoch başında zero_grad kalanı siler. Pozitif integer girdi kontrolü yok.
- **Kullanıcı etkisi:** Örn. 3 batch/accum=2 son batch metriğe katılır ama ağırlık güncellemesine katılmaz. 0/geçersiz değerler hata veya yanlış ölçek üretebilir.
- **Doğrulama durumu:** Statik akış; backward/fit/optimizer step çalıştırılmadı.
- **P0.1-C test önerisi:** Bölünen ve bölünmeyen loader uzunluğunda step sayısı, parametre deltası/kısmi pencere ölçeği; invalid accumulation değerleri.

### R10 — fit scheduler iki kez ilerletilebiliyor (P1)

- **Dosya/satır:** `hyso/models/bcam.py:295-316,390-395,467-477,530-537; hyso/callbacks/callbacks.py:49-53`.
- **Teknik açıklama:** fit scheduler nesnesini callback listesine ekler; validation sonunda doğrudan ve callback içinde tekrar step edilir. OneCycle optimizer adımı yanında epoch callback step alır. Validation yoksa epoch scheduler callbackleri çalışmaz.
- **Kullanıcı etkisi:** LR beklenenden hızlı değişebilir; OneCycle total_steps aşımı yutulabilir veya validation yokken epoch takvimi durabilir.
- **Doğrulama durumu:** Statik kontrol akışı; scheduler sayacı veya LR runtime ölçülmedi.
- **P0.1-C test önerisi:** Tüm scheduler türlerinde spy step sayısı/LR dizisi; callbacks yolu ve fit yolu, val açık/kapalı ayrı senaryolar.

### R11 — OneCycle bütçesi accumulation ve AMP update sayısından kopuk (P1)

- **Dosya/satır:** `hyso/models/bcam.py:300-304,383-395`.
- **Teknik açıklama:** steps_per_epoch=len(loader) ama OneCycle accumulation eşiğinde ilerler; AMP overflow optimizer adımını atlarsa scheduler yine çağrılır.
- **Kullanıcı etkisi:** Takvim tamamlanmayabilir veya gerçek ağırlık güncellemelerinden kopabilir; R10 fazladan callback step durumu ağırlaştırır.
- **Doğrulama durumu:** Statik; AMP/overflow çalıştırılmadı.
- **P0.1-C test önerisi:** Accumulation>1 ve simulated scaler skip ile optimizer update/scheduler sayacı eşliği; max_lr ve total_steps sınırları.

### R12 — AMP seçimi model cihazına bağlı değil (P1)

- **Dosya/satır:** `hyso/models/bcam.py:333-335,359-363; hyso/setup.cfg:20`.
- **Teknik açıklama:** AMP enabled ve autocast device_type global cuda.is_available ile seçilir. CUDA görünen makinede device=cpu modeli cuda autocast/scaler yoluna girer. torch.cuda.amp.GradScaler ve torch.amp.autocast birlikte kullanılır.
- **Kullanıcı etkisi:** CPU tercihi AMP seçimine yansımaz; karışık cihaz davranışı ve torch>=1.13 alt sınırı belirsizdir.
- **Doğrulama durumu:** Statik; CPU forward AMP kapalıydı. CPU fit AMP, CUDA ve minimum sürüm sınanmadı.
- **P0.1-C test önerisi:** CPU-only, CUDA görünen+CPU model, CUDA model × AMP on/off; dtype/device/scaler enabled ve minimum torch sürümü matrisi.

### R13 — Model taşıma ve self.device ayrışabilir (P1)

- **Dosya/satır:** `hyso/models/bcam.py:200-202,281-282; hyso/models/WithSE.py:145-146,203-215,251-270; hyso/models/fit.py:40-41`.
- **Teknik açıklama:** .to self.device alanını otomatik güncellemez. CEMO constructor cihaz seçer ama modeli taşımaz; fit/callbacks taşıma hatalarını yutar. CNNS explicit device argümanını otomatik cihazla ezer. EMA kayıt dışıdır.
- **Kullanıcı etkisi:** CPU tercihi kaybolabilir veya input/model/EMA farklı cihazda kalabilir; kök hata daha geç görünür.
- **Doğrulama durumu:** Statik; EBO/CEMO CPU constructor başarılı, CNNS ve device mismatch denenmedi.
- **P0.1-C test önerisi:** Constructor device → to → callbacks → fit sırasıyla cihazları kontrol et; CUDA bulunan ortamda explicit CPU ve EMA durumunu dahil et.

### R14 — CEMO fit use_ema gerçek kullanımını kontrol etmiyor (P1)

- **Dosya/satır:** `hyso/models/WithSE.py:183-187,251,259-270,300-305,329-337`.
- **Teknik açıklama:** Fit flag true tek başına EMA oluşturmaz; attr varsa update ve validation flagden bağımsızdır. EMA inference hatası raw modele fallback eder.
- **Kullanıcı etkisi:** Kullanıcı EMA kapalı sanırken EMA validation veya açık sanırken raw validation olabilir; model seçimi anlaşılmaz hale gelir.
- **Doğrulama durumu:** Statik; constructor EMA açık/kapalı kuruldu, fit denenmedi.
- **P0.1-C test önerisi:** Constructor flag × fit flag matrisi, update/inference çağrı sayısı; EMA hatasında fallback görünürlüğü.

### R15 — CNNS EMA validation train modunda kalabilir (P1)

- **Dosya/satır:** `hyso/models/fit.py:104-106,122,142-143,159,168-169; hyso/models/WithSE.py:24-29`.
- **Teknik açıklama:** Deepcopy o anki training modunu taşır; CNNS gölge eval/to çağırmaz. self.eval kayıt dışı EMA gölgesine yayılmaz.
- **Kullanıcı etkisi:** Validation dropout/stochastic depth veya BN istatistik güncellemesi içerebilir; küçük batchte hata veya kararsız skor olabilir.
- **Doğrulama durumu:** Statik; CNNS constructor/fit ve validation çalıştırılmadı.
- **P0.1-C test önerisi:** Gölge training bayrağı, validation boyunca BN buffer değişmezliği, çıktı determinismi ve device eşliği.

### R16 — Broad try/except kritik işlemleri gizliyor (P1)

- **Dosya/satır:** `hyso/models/WithSE.py:31-36,203-215,253-270,300-305,329-335,386-422; hyso/models/bcam.py:214-218,326-330,391-402,467-477,504-537; hyso/models/fit.py:22-25`.
- **Teknik açıklama:** Device/EMA/scheduler/save/restore hataları geniş except/pass ile yutulur. Fallbackler sınıflarda olmayan step/restore_best/on_epoch_end çağırabilir. Summary bare except kullanır. WithSE metrik/plot hatası ise mesaj yazdırır.
- **Kullanıcı etkisi:** Eğitim başarılı görünürken checkpoint, EMA, LR veya restore işlemi çalışmamış olabilir; kök neden kaybolur.
- **Doğrulama durumu:** Statik; hata enjeksiyonu yapılmadı.
- **P0.1-C test önerisi:** Failing callback/scheduler/device/EMA ile exception/log sözleşmesi; checkpoint yazma ve restore hatasının görünürlüğünü ayrı ölç.

### R17 — Paketleme ve import yüzeyi kırılgan (P1)

- **Dosya/satır:** `hyso/setup.cfg:17-29; hyso/pyproject.toml:1-3; hyso/models/bcam.py:13; hyso/models/fit.py:16; hyso/models/WithSE.py:18; hyso/callbacks/__init__.py`.
- **Teknik açıklama:** Build dosyaları rootta yok, nested discovery üst hyso paketini kapsamayabilir. torchsummary koşulsuz ama metadata dışında. WithSE __all__ tanımsız CNNSENormalWithFit içerir. Callback paketinde re-export yok.
- **Kullanıcı etkisi:** Root install, temiz declared-deps install veya wildcard import kırılabilir; geliştirici ortamı eksik deklarasyonu gizleyebilir.
- **Doğrulama durumu:** Dosya/AST ve setuptools discovery doğrulandı; wheel build/install, torchsummary kaldırma veya wildcard import denenmedi. Hazır ortam normal importu başarılı.
- **P0.1-C test önerisi:** Root/nested build ve wheel içerik kontrolü; declared-only deps temiz venv import; explicit/star import yüzeyi.

### R18 — EMA metriği ile raw checkpoint ve best restore ayrışıyor (P1)

- **Dosya/satır:** `hyso/callbacks/callbacks.py:18,28,37-40; hyso/models/bcam.py:429-435,516-524; hyso/models/WithSE.py:329-335,398-409; hyso/models/fit.py:168-169,210-216`.
- **Teknik açıklama:** Validation EMA kullanabilir ama save raw model alır. Payload yalnızca state_dict. Best restore yalnızca early_stop dalında, normal epoch bitişinde yok. EarlyStopping checkpoint callbackinden önce return edebilir.
- **Kullanıcı etkisi:** Kaydedilen ağırlık raporlanan validation skorunu üretmeyebilir; tam resume metadata yok; normal bitişte best restore beklentisi karşılanmaz.
- **Doğrulama durumu:** Statik akış; checkpoint dosyası/metrik eşliği denenmedi. R02 storage alias bağımsız runtime doğrulandı.
- **P0.1-C test önerisi:** Validation ağırlığı ile save payload eşliği; normal/erken bitiş restore sözleşmesi ve resume kapsamı.

### R19 — Tekrarlı callbacks eski state biriktiriyor (P2)

- **Dosya/satır:** `hyso/models/bcam.py:221-265,314-316; hyso/models/WithSE.py:217-246; hyso/models/fit.py:52-88`.
- **Teknik açıklama:** Callback listesi temizlenmez. Eski scheduler eski optimizera bağlı kalabilir; earlystop statei tekrar eğitimde taşınabilir. EBO/CNNS SGD caller optimizer_params sözlüğüne momentum ekler.
- **Kullanıcı etkisi:** Birden fazla scheduler/checkpoint/earlystop çalışabilir; yeni eğitim eski counter/best_loss ile başlayabilir; caller konfigürasyonu değişir.
- **Doğrulama durumu:** Statik; callbacks/fit tekrarlı çalıştırılmadı.
- **P0.1-C test önerisi:** İki callbacks ve iki fit çağrısında callback adedi, optimizer binding, earlystop reset ve dış dict yan etkisi.

## Henüz doğrulanmamış, belirli girdiye/ortama bağlı riskler

### R20 — Boş loader ve etiket şekli/tek sınıf sınırları (P2)

- **Dosya/satır:** `hyso/models/fit.py:98,112,130-147,150-151,173-195; hyso/models/bcam.py:319,338,365-378; hyso/models/WithSE.py:176,292-296`.
- **Teknik açıklama:** CNNS/EBO summary next(iter(train_loader)) ister. CNNS ortalamaları sıfıra bölme korumaz; multiclass hedefler düzleştirilmez. CEMO tek sınıfta CrossEntropyLoss kullanır; EBO/CNNS BCE yolları farklıdır.
- **Kullanıcı etkisi:** Boş/küçük split veya dış loaderda StopIteration, yanlış broadcast/accuracy ya da anlamsız binary eğitim görülebilir.
- **Doğrulama durumu:** Kontrol eksikleri statik, bu belirli girdiler çalıştırılmadı; gerçekleşmiş hata olarak sınıflanmadı.
- **P0.1-C test önerisi:** Empty train/val, N ve Nx1 integer, NxC one-hot, binary/multiclass ve küçük batch; red/return sözleşmesini açık belirle.

### R21 — Summary ve görselleştirme eğitim yan etkisi oluşturabilir (P2)

- **Dosya/satır:** `hyso/models/bcam.py:338-347,493-500; hyso/models/fit.py:20-25,112-115,195-204; hyso/models/WithSE.py:371-384`.
- **Teknik açıklama:** Summary loaderdan fazladan batch çeker ve dış torchsummary çalıştırır; RNG/BN etkisi bağımlılık/mod durumuna bağlıdır. Her validation plt.show çağırabilir; headless davranış backend bağımlıdır.
- **Kullanıcı etkisi:** Tekrarlanabilirlik, kaynak tüketimi veya otomasyonun beklemesi etkilenebilir.
- **Doğrulama durumu:** Çağrılar statik; summary implementasyonu, gerçek loader ve grafik backend senaryosu denenmedi.
- **P0.1-C test önerisi:** Summary öncesi/sonrası BN/RNG/state karşılaştırması ve headless noninteractive validation; stdout/plot çağrı sınırları.

## P0.1-B hazırlığı

P0.1-A envanteri ve CPU referansı hazır; bilinen hatalar düzeltilmeden belgelendi. Base ortam torch içermediğinden kullanılabilir başlangıç ortamı değildir; hazır torch_gpu dar baseline için yeterliydi. Tekrarlanabilir temiz dependency ortamı, minimum Python/torch matrisi ve paket build kökü kararı sonraki aşamada netleştirilmeli.

P0.1-B baseline hedefleri canonical import/imza/return, küçük model state key/shape/dtype/buffer, wrapper-prefix farkı ve strict same-config yüklemedir. Bilinen hatalı davranışın referansı ile P0.1-C düzeltme beklentisi ayrılmalı; EBO forward veya restore düzelmiş varsayılmamalı. P0.1-C test önerileri R01–R21 altındadır.

Gerçek eski checkpoint yok; tarihsel uyumluluk iddiası için dosya, constructor ve sınıf haritası gerekir. Yeni manifest bunu ikame etmez. P0.1-B veya P0.1-C'ye geçilmedi; test/kod geliştirilmedi.
