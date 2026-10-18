# HYSo P0.1-A — Repository baseline

Denetim: 18 Ekim 2026 (Europe/Istanbul). Branch: `main`. HEAD: `b74e3e2cdec527e7bb7b078f69636a27b19923fe`.
Kapsam mevcut yerel checkout'tur; uzak branch ve PyPI dağıtımı incelenmedi. HysoLLM kapsam dışıdır.

## Başlangıç durumu ve yöntem

`git status --porcelain=v1` başlangıçta boştu: çalışma ağacı ve index temiz. Başlangıç diff'leri boştu. Repo içinde ve çalışma dizininin tüm üst dizinlerinde uygulanabilir AGENTS.md bulunmadı. Üretim kodu, README, LICENSE ve paketleme dosyaları değiştirilmedi; hiçbir bağımlılık kurulmadı.

Kaynaklar UTF-8 okunup AST ile parse edildi. Kaynak olguları, runtime gözlemleri ve doğrulanmamış senaryolar ayrı tutuldu. İnternet gerekmedi: verilen deponun yerel kaynakları incelendi. Eğitim, gerçek dataset, tarihsel checkpoint, CUDA eğitim akışı veya dağıtım build'i çalıştırılmadı.

## Başlangıç dosya ağacı

17 takip edilen dosya:

```text
.gitattributes
.gitignore
LICENSE
README.md
hyso/LICENSE
hyso/README.md
hyso/__init__.py
hyso/augmentation/__init__.py
hyso/augmentation/augmentation.py
hyso/callbacks/__init__.py
hyso/callbacks/callbacks.py
hyso/models/WithSE.py
hyso/models/__init__.py
hyso/models/bcam.py
hyso/models/fit.py
hyso/pyproject.toml
hyso/setup.cfg
```

Dokuz Python kaynak dosyası vardır. Başlangıçta tests/, docs/, regression testi, pytest/tox/nox yapılandırması, CI workflow'u, lockfile veya referans fixture yoktu. .gitignore içindeki pytest/coverage kalıpları test altyapısı sayılmaz. .git yalnızca Git metadata'sıdır.

## Paketleme konumu ve metadata

**Root'ta setup.py, setup.cfg veya pyproject.toml yoktur.** Paketleme `hyso/setup.cfg:1-29` ve `hyso/pyproject.toml:1-3` altındadır. README'nin root'tan önerdiği `pip install .` için build tanımı yoktur; kurulum denenmedi.

Metadata: name=hyso, version=0.1.2, author=Huseyin Dogan, author_email=`hdgn5838@gmial.com`, description=`Hyso: Custom CNN models, SE/CBAM blocks, callbacks, and augmentations`, URL=`https://github.com/huseyin-dgn/hyso`, license=MIT, Python>=3.8; classifiers Python 3 / MIT / OS Independent. Long description `file: README.md`, içerik türü text/markdown. Build backend setuptools.build_meta; build gereksinimleri setuptools>=42 ve wheel.

`packages=find:` ve boş `where=` (setup.cfg:17,28-29) nested build köküne bağlı keşif riski taşır. Kurulu setuptools 69.0.3 ile salt okunur `find_packages('.')` → `['hyso','hyso.augmentation','hyso.callbacks','hyso.models']`; `find_packages('hyso')` → `['augmentation','callbacks','models']`. Nested build üst hyso paketini kaybedebilir. Bu discovery gözlemidir; wheel içeriği veya gerçek kurulum doğrulaması değildir.

İki README'de gösterilen bcam_model.py, bcam_fit.py, get_loaders.py, setup.py ve BCAMWithFit kaynakta yoktur. Örnek fit(scheduler=..., early_stopping=..., checkpoint_path=...) gerçek EBO imzasıyla uyuşmaz. README lisans olmadığını söylese de iki LICENSE ve metadata MIT belirtir. README iddiaları gerçek API yerine alınmadı.

## Bağımlılıklar ve gerçek kullanım

| Paket | install_requires | Kaynak kullanımı |
|---|---|---|
| torch | >=1.13 | modeller, eğitim, callback, loader |
| torchvision | >=0.14 | datasets/transforms |
| numpy | sürüm sınırı yok | bcam ve WithSE metrik dizileri |
| matplotlib | sınır yok | bcam/fit import anında; WithSE validation sırasında |
| seaborn | sınır yok | bcam/fit import anında; WithSE validation sırasında |
| scikit-learn | sınır yok | sklearn.metrics; bcam/fit erken, WithSE geç import |
| tqdm | sınır yok | eğitim ilerlemesi |
| torchsummary | **beyan edilmemiş** | bcam.py:13 ve fit.py:16 koşulsuz import |

Standart kütüphane copy EMA için kullanılır; augmentation içindeki os ve WithSE içindeki F importu kullanılmıyor. Minimum sürüm matrisi doğrulanmadı; beyan edilen torch>=1.13 ile kullanılan torch.amp.autocast yüzeyi ayrıca sınanmalı. ONNX export/inference API'si, donanım uyumluluk kontrol API'si veya harici model adaptörü kaynakta bulunmadı; gelecek ürün hedefleri mevcut özellik sayılmadı.

## İç import ilişkileri

- hyso → models, callbacks, augmentation (eager).
- models → bcam.EBO/ELO, WithSE.CEMO/EMA, fit.CNNSENormalWithFit.
- bcam → callbacks.callbacks ve WithSE.EMA.
- WithSE → callbacks.callbacks.
- fit → WithSE.CEMO/EMA ve callbacks.callbacks.
- augmentation paketi → augmentation.augmentation → models.fit.CEMO, torchvision, torch.
- callbacks/__init__.py boştur; callback sınıflarını re-export etmez.

augmentation.py:2 içindeki CEMO, fit.py'nin WithSE'den aldığı geçerli transitif alias'tır, eksik sembol değildir. Bu kullanılmayan import augmentation'ı gereksiz yere training bağımlılıklarına bağlar. Kök eager import, yalnızca callback isteyen kullanıcıya da model bağımlılıklarını yükletir.

Tam import envanteri (AST; fonksiyon içi importlar dahil):

- `hyso/__init__.py`: `L1: from . import models, callbacks, augmentation`
- `hyso/augmentation/__init__.py`: `L1: from .augmentation import build_transforms, get_loaders`
- `hyso/augmentation/augmentation.py`: `L2: from hyso.models.fit import CEMO`; `L3: from torchvision import datasets, transforms`; `L4: from torch.utils.data import DataLoader`; `L5: import torch`; `L6: import os`
- `hyso/callbacks/__init__.py`: (boş)
- `hyso/callbacks/callbacks.py`: `L1: import torch`
- `hyso/models/__init__.py`: `L1: from .bcam import EBO, ELO`; `L2: from .WithSE import CEMO, EMA`; `L3: from .fit import CNNSENormalWithFit`
- `hyso/models/bcam.py`: `L3: import torch`; `L4: import torch.nn as nn`; `L5: import torch.nn.functional as F`; `L6: import torch.optim as optim`; `L7: from tqdm import tqdm`; `L8: from ..callbacks.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback`; `L9: from .WithSE import EMA`; `L10: from sklearn.metrics import classification_report, confusion_matrix`; `L11: import seaborn as sns`; `L12: import matplotlib.pyplot as plt`; `L13: from torchsummary import summary`; `L14: import numpy as np`
- `hyso/models/fit.py`: `L1: import torch`; `L2: import torch.nn as nn`; `L3: import torch.optim as optim`; `L4: from tqdm import tqdm`; `L7: from hyso.models.WithSE import CEMO, EMA`; `L10: from hyso.callbacks.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback`; `L13: from sklearn.metrics import classification_report, confusion_matrix`; `L14: import seaborn as sns`; `L15: import matplotlib.pyplot as plt`; `L16: from torchsummary import summary`
- `hyso/models/WithSE.py`: `L5: import torch`; `L6: import torch.nn as nn`; `L7: import torch.nn.functional as F`; `L8: import copy`; `L9: import numpy as np`; `L10: from tqdm import tqdm`; `L13: from hyso.callbacks.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback`; `L372: from sklearn.metrics import classification_report, confusion_matrix`; `L375: import seaborn as sns`; `L376: import matplotlib.pyplot as plt`

## Ortam

Etkin Python `C:/Users/hdgn5/miniconda3/python.exe`, Python 3.13.12: torch, torchvision, numpy, matplotlib, seaborn, scikit-learn, torchsummary, pytest ve build yok; tqdm 4.67.3, setuptools 69.0.3 var. `import torch` ve `import hyso` yakalanan ModuleNotFoundError verdi. Bu ortam engelidir, her ortamda kaynak import hatası olduğu anlamına gelmez.

`py -0p` ve `conda env list` ile hazır `C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe` bulundu: Python 3.10.18, torch 2.5.1+cu121, torchvision 0.20.1+cu121, numpy 2.2.2, matplotlib 3.10.8, seaborn 0.13.2, scikit-learn 1.7.2, tqdm 4.67.1, torchsummary 1.5.1. Normal hyso ve hedef on sembol importu başarılı. CUDA görünür; model gözlemleri açıkça CPU'dadır. sign ve ayrı Python310 ortamlarında torch yoktur. Kurulum/indirme yapılmadı.

## Komut günlüğü

PowerShell, repo kökü. Birleşik araç çağrılarındaki bağımsız komutlar ayrılmıştır. Python içinde exception yakalandığında süreç kodunun 0 olması API başarısı değildir. `<gpu-python>` yukarıdaki torch_gpu executable yoludur; inline Python gövdelerinin yaptığı işlemler belirtilmiştir.

| Komut / işlem | Amaç | Çıkış kodu | Sonuç |
|---|---|---|---|
| Get-Content -LiteralPath <attachment> | Görevi okuma | 0 | İlk encoding bozuk; -Encoding UTF8 ile yeniden okundu |
| rg --files -g AGENTS.md -g '!node_modules' -g '!vendor' | Yönerge arama | 1 | Eşleşme yok; ilk birleşik çağrının kodu bu nedenle 1 |
| git status --porcelain=v1; git branch --show-current; git rev-parse HEAD | Başlangıç | Her biri 0 | Temiz / main / rapor HEAD'i |
| rg --files; git ls-files; Get-ChildItem -Force | Dosya ağacı | Her biri 0 | 17 takip edilen dosya |
| Get-Content -Encoding UTF8 <kaynaklar> + satır numarası döngüleri | Tüm kaynak/metadata/README okuma | 0 | Statik envanter |
| python --version | Etkin Python | 0 | 3.13.12 |
| python -B - + importlib.metadata ve import denemeleri | Bağımlılıklar | 0 | Yakalanan torch/hyso import hatası |
| python -B - + ast.parse/unparse ve find_packages | İmza/import/keşif | 0 | Dokuz dosya parse edildi; nested keşif riski |
| python -B - + pathlib üst dizin AGENTS/hash/checkpoint taraması | Yönerge ve ağırlık arama | 0 | Yönerge/checkpoint yok |
| py -0p; conda env list; Get-Command python,python3,py,conda | Hazır ortamlar | Her biri 0 | torch_gpu bulundu |
| Üç hazır Python ile -B - + metadata sorgusu | Sürümler | Her biri 0 | torch_gpu hazır; diğer ikisinde torch yok |
| <gpu-python> -B - + normal canonical importlar | Paket importu | 0 | hyso ve on hedef API başarılı |
| <gpu-python> -B - + küçük CPU constructor/state/strict load/forward gözlemleri | Manifest | 0 | Altı konfigürasyon; ayrıntı JSON/checkpoint raporunda |
| git ls-files '*.pth' '*.pt' '*.ckpt' '*.onnx' + pathlib taraması | Eski checkpoint arama | 0 | Dosya yok |
| python -B - + pathlib/AST rapor üretimi | Dört rapor | 0 | İlk yazımda native pipeline Türkçe karakter kaybı; $OutputEncoding UTF8 ile raporlar yeniden üretildi |
| python -B - + JSON/hash/prefix doğrulaması | Yapısal QA | 0 | Altı model, dokuz kaynak hash'i ve EBO prefix eşleşmesi başarılı |
| git ls-files --eol | Satır sonu farklarını ayırma | 0 | HEAD LF, bazı checkout dosyaları CRLF |
| git hash-object --path=<p> <p> ile git rev-parse HEAD:<p> karşılaştırması | Tüm takip edilen dosyalar | 0 | Git clean/EOL filtreleri sonrası 17 dosya eşleşti |
| git diff --exit-code; git diff --cached --exit-code | Çalışma ağacı/index | Her biri 0 | Üretim dahil tracked diff yok |
| git status --short --untracked-files=all; git rev-parse HEAD | Son durum | 0 | Yalnızca izinli beş yeni dosya, HEAD aynı |

Python -B bytecode yazımını engelledi. Inline denemeler kalıcı test dosyası/regression suite değildir. fit, optimizer adımı, gerçek dataset, checkpoint dosyası kaydı, CUDA eğitimi ve wheel build çalıştırılmadı.

## Bütünlük ve bitiş

Manifest dokuz kaynak dosyanın raw SHA256 değerlerini saklar; son kontrolde hepsi değişmeden kaldı. Ayrıca 17 tracked dosyanın Git clean/EOL filtreleri sonrası blob kimliği HEAD ile eşleşti. CRLF checkout nedeniyle raw byte'lar HEAD blob'uyla aynı sayılmadı; filtreli karşılaştırma ve raw kaynak hash kontrolü ayrı kullanıldı.

Son status beklenen beş untracked çıktıdır: docs/audit altındaki dört .md raporu ve tests/fixtures/model_state_manifest.json. Index ve tracked diff boş; branch/HEAD aynı. Tracked dosyalarda değişiklik yok; önceden mevcut ignored dosyaların sınırları aşağıda açıklanmıştır. Commit/push/merge/rebase/tag/release komutu çalıştırılmadı. Son kalite kontrol sonuçları aşağıya eklenir.

## Yerelde bulunan ignored dağıtım ve cache dosyaları

Son tam disk taraması, ilk normal git status/rg envanterinin göstermediği **16 ignored dosya** buldu. Bunlar önceden mevcut yerel dist/, hyso.egg-info/ ve __pycache__/ dosyalarıdır; aşağıdaki LastWriteTime tarihleri denetimden eskidir. Hiçbiri silinmedi, üzerine yazılmadı veya bu görevde oluşturulmadı. Python komutlarının tümü -B kullandı; build çalıştırılmadı. İlk durum temiz ifadesi Git'in normal tracked/untracked görünümüne aittir, ignored dosyaların bulunmadığı anlamına gelmez.

| Yerel ignored dosya | Boyut (byte) | Son yazım (dosya sistemi) |
|---|---:|---|
| `hyso/augmentation/__pycache__/__init__.cpython-310.pyc` | 240 | 2025-10-03T15:42:46 |
| `hyso/augmentation/__pycache__/augmentation.cpython-310.pyc` | 2695 | 2025-10-03T15:42:46 |
| `hyso/callbacks/__pycache__/__init__.cpython-310.pyc` | 133 | 2025-10-03T15:42:44 |
| `hyso/callbacks/__pycache__/callbacks.cpython-310.pyc` | 2414 | 2025-10-03T15:42:44 |
| `hyso/dist/hyso-0.1.2-py3-none-any.whl` | 18505 | 2025-10-04T00:05:45 |
| `hyso/dist/hyso-0.1.2.tar.gz` | 14471 | 2025-10-04T00:05:41 |
| `hyso/hyso.egg-info/PKG-INFO` | 3554 | 2025-10-04T00:05:41 |
| `hyso/hyso.egg-info/SOURCES.txt` | 345 | 2025-10-04T00:05:41 |
| `hyso/hyso.egg-info/dependency_links.txt` | 1 | 2025-10-04T00:05:41 |
| `hyso/hyso.egg-info/requires.txt` | 73 | 2025-10-04T00:05:41 |
| `hyso/hyso.egg-info/top_level.txt` | 30 | 2025-10-04T00:05:41 |
| `hyso/models/__pycache__/WithBCAM.cpython-310.pyc` | 5127 | 2025-10-03T13:58:35 |
| `hyso/models/__pycache__/WithSE.cpython-310.pyc` | 10552 | 2025-10-03T15:16:42 |
| `hyso/models/__pycache__/__init__.cpython-310.pyc` | 271 | 2025-10-03T14:00:34 |
| `hyso/models/__pycache__/bcam.cpython-310.pyc` | 13124 | 2025-10-03T15:42:42 |
| `hyso/models/__pycache__/fit.cpython-310.pyc` | 5514 | 2025-10-03T14:05:52 |

Wheel adı hyso-0.1.2-py3-none-any.whl, sdist adı hyso-0.1.2.tar.gz'dir. Salt okunur zipfile ile wheel içeriğini açma denemesi PermissionError (Errno 13) verdi; aynı inline komutta planlanan sdist listelemesi bu nedenle çalışmadı. Dosyaların paket içeriği/metadata'sı doğrulanmadı; adlarından HEAD'e ait oldukları varsayılmadı. Kaynak metadata incelemesi tracked setup.cfg/pyproject.toml'a dayanır. Bu ignored dağıtımlar tarihsel model checkpoint'i değildir. Başlangıçta ignored dosyaların byte hash'i alınmadığı ve wheel okuması reddedildiği için bunlar hakkında byte düzeyinde önce/sonra eşitlik iddiası yoktur; herhangi bir yazma işlemi yapılmadı.

Ek komutlar: tam pathlib dosya kümesi kontrolü ilk denemede çıkış 1 verdi (gözden kaçan ignored dosyaları ortaya çıkardı); Get-ChildItem -Recurse -Force ve git status --short --ignored --untracked-files=all başarılıydı. Wheel okuma inline Python komutu çıkış 1 (PermissionError). Son kontrol normal çıktı kümesiyle ignored kümesini ayrı doğrulayacak şekilde düzeltildi. Bunlar üretim kodu veya test başarısızlığı değildir.

### Tamamlanan son kalite kontrolü

Son inline doğrulama (`python -B -`; pathlib/json/ast/hashlib/subprocess), çıkış 0: dört rapor UTF-8 Türkçe karakter kontrolü; altı model manifest yapısı; iki EBO prefix karşılaştırması; dokuz raw kaynak hash değeri; 17 Git-filtreli HEAD blob eşitliği başarılı. Tam dosya kümesi 38 = 17 tracked + 16 önceden mevcut ignored + izinli 5 çıktı. Ignored dosyaların son yazım tarihleri denetimden eskidir; byte hash başlangıcı alınmadığından bu dosyalar için byte eşitliği iddiası yoktur.

`git diff --check`, `git diff --exit-code`, `git diff --cached --exit-code` ayrı ayrı 0. Son `git status --short --untracked-files=all` (çıkış 0):

```text
?? docs/audit/checkpoint-compatibility.md
?? docs/audit/public-api-inventory.md
?? docs/audit/repository-baseline.md
?? docs/audit/risk-inventory.md
?? tests/fixtures/model_state_manifest.json
```

Branch main ve HEAD b74e3e2cdec527e7bb7b078f69636a27b19923fe değişmedi. Üretim/index değişikliği, test dosyası veya ağırlık dosyası üretimi yok. Commit/push yapılmadı. P0.1-A tamamlandı; P0.1-B başlatılmadı.
