> Historical document: the web product was retired in stage 13.5.6. Do not use the launch/API instructions below as an active product guide. See [Python-first transition](../reports/stage13-5-6-python-first-transition.md).

# Studio local API

Studio binds IPv4 `127.0.0.1`. The port is configurable. Windows uses exclusive socket binding to detect port conflicts reliably. The API has no LAN bind option, plugin loader, shell command endpoint, user-supplied filesystem locator or HTTP subprocess dispatcher.

The server accepts the exact numeric loopback Host and same Origin. Cross-site browser requests and CORS preflights are rejected. API requests require `X-HYSo-API` equal to `1` and the per-launch `X-HYSo-Token` embedded in the served HTML meta element. These checks protect the local browser boundary. The token changes when the server restarts. Frontend startup checks both Studio and API versions.

Reborn additionally requires `workflow_contract=3` from health before enabling POST requests. A mismatch leaves saved records intact and shows restart instructions. The five served assets are frozen at server startup, so updating source requires restarting Python and reloading the browser together.

Static serving uses five fixed asset names. JSON bodies are limited to 2 MiB, nesting 20, strings 64 KiB and containers 65536 items. Duplicate keys, nonfinite numbers, transfer encodings, unexpected fields and duplicate Content-Length are rejected. Core configuration validation retains its tighter existing bounds. At most sixteen HTTP handlers run concurrently. Responses have no-store, nosniff, same-origin CSP, frame-ancestor and referrer protections.

## Endpoints

| Method | Path | Result |
| --- | --- | --- |
| GET | `/api/health` | API and Studio version, status |
| GET | `/api/overview` | Real model/version/result/comparison totals and bounded recent summaries |
| GET | `/api/contract` | Machine-readable request envelopes, methods, auth and job states |
| GET | `/api/catalog` | Actual Registry and fusion form metadata |
| GET | `/api/capabilities` | Capability/source/validation/test inventory |
| GET | `/api/models` | Saved drafts with revisions |
| GET / POST | `/api/versions` | List immutable versions or publish a valid saved revision |
| POST | `/api/architectures/compare` | Compare 2..5 saved draft/version references allocation-free |
| POST | `/api/job-preflight` | Validate setup and data without creating tensors or queueing a job |
| POST | `/api/models` | Save draft with expected revision |
| POST | `/api/preview` | Core allocation-free preflight and config hash |
| POST | `/api/snippet` | Core Python snippet and execution eligibility |
| GET / POST | `/api/settings` | Setting definitions, overrides and validated updates |
| GET | `/api/records` | Exact hash/version/model/kind/date/run/experiment filters |
| POST | `/api/records` | Closed-format result import with explicit provenance |
| POST | `/api/recommendations` | Current diagnostics, copied candidate validation and native Advisor |
| POST | `/api/report` | Bounded direct report service |
| GET / POST | `/api/jobs` | List (optional id or summary=1) or explicitly submit durable CPU jobs |
| POST | `/api/cancel` | Safely cancel a queued job by ID |
| GET | `/api/experiments` | Studio-owned native ExperimentManager records |
| POST | `/api/comparison` | Compare 2..8 distinct retained record IDs with conditions and native metrics |

Error responses use HTTP 400 for invalid input, 403 for browser-boundary rejection, 404 for unknown resources and 500 for an unexpected internal operation. Saturated HTTP admission returns 503. The standalone Studio CLI has its documented startup exit behavior. Existing HYSo CLI commands retain their existing `hyso.cli` schema1 envelopes and exit codes.

## Config and job examples

The API accepts the same graph configuration used by the public Python API.

```python
from hyso.models import GraphModelBuilder

config = (
    GraphModelBuilder({'image': [3, 8, 8]})
    .add('stem', 'conv2d', ['image'], in_channels=3,
         out_channels=4, kernel_size=3, padding=1)
    .outputs(features='stem')
    .to_config()
)
request = {'config': config, 'disabled_nodes': []}
```

POST this request to `/api/preview` with the local session headers. The returned report includes Core node geometry, exact/conservative parameter information, attention cost, activation estimates, SCC cycle membership and the configuration SHA-256.

Job envelopes contain `type`, `request_id`, `config`, optional `disabled_nodes` and type-specific `options`. Supported types are validate, build, test, train, benchmark, evaluate, report and analyze. Reusing an accepted job's ID with identical content returns its existing job. Reusing that ID with different content fails. An identical active operation submitted with another ID joins the original job and returns its canonical `request_id`. Retain that returned ID for retries. The UI retains retry keys and guards double activation. Worker dispatch failure persists a FAILED record. There is one worker and at most eight queued/running jobs.

| Job | Options |
| --- | --- |
| validate/analyze | Empty object |
| build | Optional `parameter_count_only=true`: explicit <=1M Core-bound CPU count, no forward pass |
| test | Optional `tests`: distinct nonempty subset of `forward`, `backward`, `serialization`; omitted retains all three |
| train | `features`, `targets`, `training`, required graph `head_mapping` |
| benchmark | `features`, required graph `head_mapping` |
| evaluate | `targets`, `predictions`, `class_names`, `split` |
| report | Optional `language`, `format`, `include`, `filters`, `precision`, `metrics`, `record_kinds`, `include_provenance`, `include_warnings` |

Training accepts existing TrainingConfig optimizer/scheduler fields through its closed Studio whitelist. The interactive profile is CPU fp32, up to five epochs and batch size 32. It accepts finite user-supplied NC/NCHW data up to 65536 elements. Graphs require one input. Training uses an explicitly selected classification head through the existing TrainingSession. BenchmarkRunner consumes an explicitly selected graph output. All configured graph outputs are evaluated. Benchmark uses one warmup, three measured iterations and one repeat with fresh initialization.

For graph training, options include `head_mapping={"output":"logits","loss":"cross_entropy"}` with `training.task="multiclass"`, or `loss="bce_with_logits"` and `task="binary"`. The selected named output must reference a compatible `classification_head`. Training targets are integer multiclass indices or binary hard labels. Graph benchmark uses `head_mapping={"output":"features"}`. Missing or incompatible mappings are rejected before queueing. One selected loss is supported per run. Multi-loss optimization and multi-input dataset adapters are unavailable. Provenance retains the mapping and exact original graph hash.

The test job explicitly uses fresh seed-0 initialization and synthetic zero inputs for eval-mode forward/backward and strict state_dict roundtrip. Its record identifies that fixture and makes no accuracy claim. Evaluation consumes supplied predictions through the existing classification evaluator. It records the supplied-prediction source.

Short runtime jobs require concrete activation geometry from Core preflight. Complete `modular_cnn` schema1 configurations have no concrete input geometry in this contract, so their Studio runtime test/train/benchmark controls are unavailable. Their Core JSON, preview and build paths remain available.

Jobs persist ID, type, timestamps, status, config hash, error and result references. Unknown progress is null. Runtime tests persist selected tests, actual phase and completed case observations; these are not an elapsed-time percentage. Training's existing epoch observer records completed/requested epochs and observed learning rate. It fabricates no percentage. Cancellation is available while QUEUED. A restarted application marks formerly queued/running records INTERRUPTED. Existing jobs are never automatically resumed. ExperimentManager handles remain worker/request-local. Native Manager run events describe result retention. Studio job timestamps describe execution.

An operating-system file lease gives each workspace one Studio process owner. A second launch using another port must select a different workspace. The OS releases the lease when its owner exits. The fixed lock file can remain present between launches. This prevents a second job manager from interrupting active job records or dispatching duplicate work in the same store.

## Türkçe API ve iş sözleşmesi

Sunucu yalnız `127.0.0.1` üzerinde çalışır. API çağrıları oturum başına değişen `X-HYSo-Token` ve `X-HYSo-API: 1` başlıklarını ister. Host/Origin aynı yerel oturumu göstermelidir. İstekler 2 MiB, nesting 20, string 64 KiB ve container başına 65536 elemanla sınırlıdır. Yinelenen anahtar, sonlu olmayan sayı, beklenmeyen alan ve aktarım kodlaması reddedilir. Sabit varlık yolları ve kapalı JSON şemaları kullanılır.

`/api/preview` allocation-free Core ön doğrulaması, `/api/models` taslak kaydı, `/api/records` açık provenance ile sonuç aktarımı, `/api/recommendations` kanıt analizi, `/api/report` rapor üretimi ve `/api/jobs` açık CPU işi başlatımı sağlar. `/api/comparison`, `record_ids` içinde 2–8 ayrı kayıt ve isteğe bağlı `language` alır. Yol ve zarf listesi `/api/contract` içinde makine tarafından okunabilir.

Graph eğitiminde tek giriş ve tek açık classification loss/head seçimi desteklenir. `head_mapping.output` adlandırılmış çıkıştır. Multiclass için `cross_entropy`, binary için `bce_with_logits` seçilir. Benchmark yalnız açık çıkış seçimi ister. Tüm graph çıkışları hesaplanır. Kaynak graph hash ve eşleme sonuç provenance alanında korunur. Çoklu loss, çoklu girişli dataset ve checkpoint/full resume bu adaptörün kapsamı dışındadır.

İş kimliği aynı içerikle tekrar kullanıldığında aynı iş döner. Aynı aktif işlem farklı istek kimliğiyle gelirse mevcut işe bağlanır. Bir worker ve en çok sekiz aktif/kuyruk işi bulunur. Kuyruktaki iş iptal edilebilir. Yeniden başlatma yarım işleri INTERRUPTED yapar. Gerçek epoch gözlemi saklanır, bilinmeyen ilerleme null kalır. Workspace kilidi ikinci süreç sahibini engeller.

Model düzenlemede büyük integer değerleri BigInt/decimal metin üzerinden kayıpsız sayısal JSON olarak taşınır. Native kanıt yüklemesi özgün sayı literal'larını korur. Üretilen API yanıtlarının browser ayrıştırıcısı 32 MiB metin, 16 MiB string ve derinlik 64 sınırı kullanır. İstek giriş sınırları değişmez. Büyük rapor metni istek string sınırına tabi değildir.

## Model context and setup

Jobs and setup accept optional `model_ref={kind,id,revision}`, where kind is `model` or `version`. The reference must resolve to the submitted configuration and disabled-node snapshot. Stale drafts fail. Accepted request retries remain idempotent even after later draft edits. Jobs persist the frozen model snapshot and provenance. Publication accepts `model_id`, integer `revision` and `request_id`; it appends a validated immutable snapshot. There is no version update endpoint.

`/api/job-preflight` uses the normal job admission path but creates no job or tensors. Training/data fields are closed and checked against the Core input geometry and explicit classification output/loss. Execution still requires POST `/api/jobs` with a request ID. Data source name/dataset/split are optional declarations; local feature/target digests supplement result-comparison conditions.

V2 result filters also accept `model_revision` together with `model_id` for a saved draft. This filter excludes version records; use `version_id` for an immutable version. Matching architecture hash alone does not relabel records from another saved model/revision. Architecture comparison attaches exact CPU build counts only when hash, reference kind/ID and revision match. Its per-column count state is EXACT, ESTIMATED or UNAVAILABLE with bilingual reason and provenance. Preview and comparison do not allocate models.

Unsaved execution snapshots can use `model_ref={kind:"draft",id,revision,name}`. These are explicitly client-labelled snapshots, not validated saved references. The job freezes their configuration and retains provenance under the separate `model_id="draft/"+id` namespace and string revision. This does not save or overwrite a model. Changing the active unsaved draft cannot attach the previous draft's run, even when architecture hashes match. Requests without model_ref remain supported for existing API callers; their records remain hash-bound and available in the general record API/comparison.

The short test/train/benchmark profile requires <=1M Core parameter bound and <=1M estimated activation elements. Test fixtures are admitted at <=65536 input elements before constructing the model. Runtime tests retain `selected_tests` and `prerequisites`; only selected checks produce cases with individually measured durations. Backward sums eval outputs and checks finite observed gradients; it is not a classification loss test. Serialization strictly reloads actual weights into a second fresh model. All remain CPU-only explicit jobs.
