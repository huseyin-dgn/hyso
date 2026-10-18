# HYSo P0.1-A — Model/checkpoint uyumluluğu

Branch main; HEAD `b74e3e2cdec527e7bb7b078f69636a27b19923fe`; 18 Ekim 2026. Referans: `tests/fixtures/model_state_manifest.json`.

## Kanıt sınırı

**Manifest yeni constructor'ların yapısal referansıdır; tarihsel checkpoint değildir.** Takip edilen dosyalarda ve .git hariç çalışma ağacı taramasında .pth/.pt/.ckpt/.onnx yoktur. Eski checkpoint yüklenmedi. Tensor değerleri veya ağırlık dosyaları kaydedilmedi.

Base Python'da torch yoktu; hazır `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe` kullanıldı: Python 3.10.18, torch 2.5.1+cu121. Normal hyso importu başarılı. CUDA görünür olsa da tüm model denemeleri CPU'da, eval modunda, seed=0, thread=1; giriş [2,3,16,16] float32. EBO/CEMO device='cpu' ile kuruldu; tüm modeller .cpu() ile taşındı. Eğitim, optimizer adımı ve AMP çalıştırılmadı.

Altı konfigürasyonda ikinci taze eş konfigürasyona `clone.load_state_dict(state, strict=True)` başarılı; missing/unexpected listeleri boş. Bu aynı kaynak ve konfigürasyonda key/shape yüklenebilirliğidir; tarihsel veya sürümler arası uyumluluk değildir. Yükleme sonrası ayrı çıktı eşitliği ölçülmedi. strict=False hiçbir aşamada kullanılmadı.

## Kesin constructor konfigürasyonları

ELO/EBO tabanı: input_channels=3, num_classes=2, conv_channels=[4,8], residual_per_block=1, p_drop=0.0, att_reduction=2, norm_type='auto', use_depthwise=False. EBO ek device='cpu'. Batch/depthwise varyantında norm_type='batch' ve use_depthwise=True.

CEMO: input_channels=3, num_classes=2, conv_channels=[4,8], residuals_per_stage=1, se_reduction=2, p_drop=0.0, dropout_fc=0.0, use_ema=False, ema_decay=0.999, device='cpu'. CEMO_ema yalnızca use_ema=True ile farklıdır. Bunlar üretim default'larına öneri değildir.

| Manifest ID | State girdisi | Parameter tensor | Buffer tensor | Parameter elemanı | CPU forward |
|---|---:|---:|---:|---:|---|
| ELO_auto | 45 | 42 | 3 | 635 | [2,2] logits |
| EBO_auto | 45 | 42 | 3 | 635 | Wrapper NotImplementedError; child [2,2] |
| ELO_batch_depthwise | 78 | 48 | 30 | 628 | [2,2] logits |
| EBO_batch_depthwise | 78 | 48 | 30 | 628 | Wrapper NotImplementedError; child [2,2] |
| CEMO | 58 | 34 | 24 | 375 | [2,2] logits |
| CEMO_ema | 58 | 34 | 24 | 375 | [2,2] logits |

CEMO_ema parent sayıları gölgeyi içermez; gölge ayrıca 58 state girdisi taşır. Bu küçük modellerden eğitim kalitesi veya performans/benchmark sonucu çıkarılmadı.

## Manifest şeması

schema_version=1; HEAD/audit_date/provenance; interpreter, bağımlılık sürümleri, device, seed, thread; dokuz kaynağın raw SHA256 değerleri. Her model için canonical import yolu, tam constructor, tüm named_modules hiyerarşisi/türü/training bayrağı, sıralı state_dict anahtarları, shape/dtype/parameter-buffer ayrımı/requires_grad, toplamlar, strict load sonucu ve forward sonucu vardır. EBO wrapper hatası child başarısından ayrı yazıldı. EMA kayıt durumu ayrıca kaydedildi.

Parameter/buffer sınıflaması named_parameters/named_buffers adlarıyla yapıldı. Parametreler ve floating buffer'lar float32; BatchNorm num_batches_tracked scalar buffer'ları int64 ve shape=[]'dir. Buffer'lar requires_grad=False. JSON tensor değerleri içermez.

observations alanı EarlyStopping storage alias gözlemi, EMA parameter/buffer gözlemi ve basic transform tür listesini içerir. Bunlar eğitim/gerçek görüntü/tarihsel checkpoint testi değildir.

## Hiyerarşi ve key aileleri

| Model | Üst hiyerarşi | State aileleri |
|---|---|---|
| ELO | stem, blocks:ModuleList, global_pool, fc_norm, head | stem.*, blocks.<flat_index>.{conv1,bn1,conv2,bn2,conv3,bn3,shortcut,attention}.*, fc_norm.*, head.0.*, head.3.* |
| EBO | model → ELO | Tüm ELO yolları model. ile başlar |
| CEMO | stem, stages:ModuleList[Sequential], global_pool, dropout_fc, fc, criterion | stem.*, stages.<stage>.<block>.{conv1,bn1,dwconv,bn2,conv_pw,bn3,se,shortcut}.*, fc.* |

ELO auto/group residual normları GroupNorm'dur; running buffer'ları yok, affine weight/bias vardır. Stem her zaman BatchNorm: auto referansta da üç buffer vardır. Batch varyant residual normlara BatchNorm buffer'ları ekler. Depthwise durumda conv2/bn2 Identity olur; dw_seq.0/.1/.3/.4 state taşır. CEMO stem/blok normları BatchNorm'dur. CEMO'da ilk [4→4] shortcut Identity, [4→8] blok projeksiyondur. Pooling, aktivasyon, dropout ve varsayılan criterion state taşımaz, fakat hiyerarşide kaydedilmiştir.

Somut örnekler; tam anahtar/shape/dtype/kind listesi JSON'dadır:

- **ELO_auto**: `stem.0.weight` [4, 3, 3, 3], torch.float32, parameter; `stem.1.running_mean` [4], torch.float32, buffer; `stem.1.num_batches_tracked` [], torch.int64, buffer; `head.3.weight` [2, 4], torch.float32, parameter.
- **EBO_auto**: `model.stem.0.weight` [4, 3, 3, 3], torch.float32, parameter; `model.head.3.weight` [2, 4], torch.float32, parameter.
- **ELO_batch_depthwise**: `stem.0.weight` [4, 3, 3, 3], torch.float32, parameter; `stem.1.running_mean` [4], torch.float32, buffer; `stem.1.num_batches_tracked` [], torch.int64, buffer; `blocks.0.dw_seq.0.weight` [1, 1, 3, 3], torch.float32, parameter; `head.3.weight` [2, 4], torch.float32, parameter.
- **CEMO**: `stem.0.weight` [4, 3, 3, 3], torch.float32, parameter; `stem.1.running_mean` [4], torch.float32, buffer; `stem.1.num_batches_tracked` [], torch.int64, buffer; `stages.1.0.conv_pw.weight` [8, 2, 1, 1], torch.float32, parameter; `fc.weight` [2, 8], torch.float32, parameter.

## Prefix ve load_state_dict sözleşmesi

EBO.state_dict() ile EBO.model.state_dict()/ELO state'i farklı key uzayındadır. Her iki EBO'da wrapper state'i child'a strict=True ile yükleme beklenen missing/unexpected anahtarlarla RuntimeError verdi; tam mesajlar JSON'dadır. EBO key'lerinden model. çıkarınca eş ELO referansının key/shape/dtype/kind listesi birebir eşleşti. Prefix göçü veya üretim düzeltmesi uygulanmadı.

bcam.py:254,524 model-level checkpoint callback'ine child ELO verir. callbacks.py:40 yalnızca verilen modelin state_dict'ini yazar. Bu nedenle fit checkpoint'i prefixesizdir; kullanıcı wrapper.state_dict() kaydederse prefix'lidir. Kaydetme/yükleme hedefini karıştırmak strict hatasına yol açabilir. CEMO WithSE.py:409 ve CNNS fit.py:216 parent state verir. Gerçek dosya kaydı bu denetimde çalıştırılmadı.

EarlyStopping.restore_weights (callbacks.py:28) verilen modelde load_state_dict kullanır; strict varsayılandır. Restore akışı çalıştırılmadı; snapshot alias hatası ayrı CPU gözleminde doğrulandı. strict=False eksik/fazla key'leri kabul ederek uyumluluk kanıtı sayılmadı; shape farkları için de genel göç çözümü değildir.

Checkpoint payload'ında constructor, sınıf indeksi haritası, optimizer/scheduler/scaler/epoch/RNG durumu yoktur. Raw parent state EMA gölgesini içermez. Bu payload eğitime eksiksiz devam etme referansı sayılmaz.

## Mimari değişiklik riskleri

- input_channels stem giriş boyutunu; num_classes ELO head.3 veya CEMO fc boyutunu değiştirir.
- conv_channels, blok/stage sayısı kernel/norm/attention shape'lerini ve flat ELO blok / CEMO stage-blok yollarını değiştirir. Shortcut eklemek/çıkarmak state key'lerini etkiler.
- norm_type BatchNorm buffer varlığını; use_depthwise key adlarını/konvolüsyon shape'lerini değiştirir. Varyantlar aynı şema değildir.
- att_reduction/se_reduction ara boyutu max(1, channels//reduction) ile belirler; her değer değişimi shape değiştirmez, değiştirdiğinde eski ağırlıklar yüklenemez.
- model/blocks/stages/head alanlarını yeniden adlandırmak key uzayını bozar. İsimler aynı kalsa bile stride/norm/forward mantığını değiştirmek strict yüklemeyi geçip çıktı semantiğini bozabilir.
- p_drop, dropout ve EMA decay genellikle state şemasını değiştirmez; şema eşitliği davranış eşitliği değildir.
- EMA normal Python nesnesidir: parent state'te ema.* yoktur; parent to/eval traversal'ı gölgeyi otomatik kapsamaz. use_ema=True/False CEMO parent state şeması aynıdır. EMA buffer'ları update ile değişmedi; validation ile kaydedilen raw ağırlıklar ayrışabilir.

## Yeniden üretme ve sonraki aşama

Yukarıdaki GPU ortam executable'ı -B ile kullanılır; normal canonical importlardan sonra tam altı config CPU/eval kurulur. named_parameters/named_buffers/named_modules/state_dict okunur; aynı config ikinci modele strict yüklenir. Forward no_grad altında CPU [2,3,16,16]; EBO wrapper ve child ayrı gözlenir. Kaynak hash'leri ve ortam sürümleri JSON'dadır; rastgele tensor değerleri yazılmaz.

P0.1-B için yapısal referanslar hazırdır. Tarihsel uyumluluk için gerçek eski checkpoint ile onun constructor ve sınıf haritası gerekir; yeni manifest bunun yerine geçirilemez. Minimum sürüm ve dağıtılmış wheel uyumluluğu ayrıca sınanmalıdır. Bu aşamada P0.1-B testlerine geçilmedi.

Yerel paket arşivi notu: normal Git görünümünün dışında ignored hyso/dist wheel/sdist ve egg-info dosyaları da bulundu. Bunlar checkpoint değildir. Wheel okuma PermissionError verdiği için paket içeriği ve hangi kaynaktan üretildiği doğrulanmadı; bu raporun referansı yalnızca mevcut tracked kaynak ve manifesttir.
