> Historical document: the web product was retired in stage 13.5.6. Do not use the launch/API instructions below as an active product guide. See [Python-first transition](../reports/stage13-5-6-python-first-transition.md).

# Studio model editor

The graph editor uses the existing `hyso.model_config` schema3 contract. Backend `GraphModelBuilder.preflight` owns tensor shapes, counts, budgets, diagnostics and topological order. The frontend stores node positions independently of the model configuration.

## Edit a graph

The untouched workspace contains a minimal input/identity graph. Create a new model asks for its name, image channels, height, width and class count, then opens editable convolution/classification layers. Neither path includes measured results or trained weights. Use search, category filter or drag/drop to add components. Select a node to edit its name, ordered connections and typed constructor fields. Arrays and pairs use semicolon-separated values; nested attention uses controls. More information explains Core bounds in natural language; exact metadata is separately expandable.

Click an output port and then a target input port to connect. Unary components replace their single reference. Fusion components append an ordered reference. The inspector exposes the full ordered list for removing or rearranging references. External skips and projection use explicit edges and ordinary projection nodes.

Use input descriptors to select NC or NCHW dimensions without batch. Named outputs map safe output names to existing input or node IDs. Multi-output configurations retain the public builder's dictionary output contract.

Node renaming updates references and named outputs. Deleting a node retains dangling references for explicit repair. Duplicating a node creates a new stable ID and preserves its original input references. Parameter edits leave unrelated nodes intact. A disabled node retains its connections and produces a blocking `STUDIO_NODE_DISABLED` diagnostic.

Background drag pans the viewport. Zoom controls, wheel zoom and Fit graph adjust the viewport. Undo and redo retain up to 80 editor snapshots. Ctrl+Z and Ctrl+Shift+Z work outside text inputs. Node selection supports focus and Enter. Arrow keys move a focused node by ten units, or forty with Shift. Real Chromium tests exercise dragging, port edits, keyboard focus and reduced motion. This is a targeted accessibility check.

## Persistence and interchange

Nodes can be moved again after their initial placement. Pointer movement uses canvas-relative coordinates divided by the current zoom; background pan does not alter model coordinates. Wheel zoom stays anchored at the pointer. Inspector selection opens after a click completes rather than changing canvas layout during a drag. Pointer up/cancel releases capture. One drag creates one undo snapshot. Saved positions retain their full numeric precision; rendered CSS may display rounded decimal coordinates. Validated versions permit inspection and viewport movement but cannot mutate saved node positions.

Save draft persists the configuration, positions, disabled nodes, name and revision in Studio SQLite. Revision checks reject a stale overwrite. Import and Export JSON preserve the public Core configuration. Draft bundle export additionally retains positions and disabled-node state in `hyso.studio.draft` schema1. Imported bundles open as new drafts.

Create validated version requires full successful Core preflight and an unchanged saved revision. Versions retain their configuration, hash, source draft revision and numbered identity in an append-only store. Repeated accepted publication requests return the same version. Reopening a version disables editing; Edit a copy creates a new draft. Later draft edits cannot overwrite a version or relabel its jobs. Jobs retain the selected model snapshot and version provenance. Moving a version's canvas viewport does not mutate its architecture.

Code and files contains Core JSON, the draft bundle and executable Python export. Python code uses the real builder API and fresh initialization. Build PyTorch model explicitly runs the Core build job. Real Chromium tests obtain Python from an edited visual graph and execute forward, backward and strict state_dict roundtrip. Architecture, trained weights and checkpoints are distinct artifacts.

The inspector can be closed to widen the canvas; on narrow screens it becomes a bottom drawer. Show dimensions reveals the Core shape/count presentation. Basic node cards omit rank/layout/dtype metadata. Technical details retain complete Core diagnostics and the source configuration.

Drafts receive a unique model identity before their first save. MODEL and NODE preference owners use that identity. Loading a saved model resolves its persisted node overrides. Saving a draft reconciles existing node overrides with the edited enable state. RUN device is read-only under the Studio CPU profile.

Schema1 and schema2 configurations can be imported, previewed and exported through JSON. The visual node editor targets schema3. The editor requires structural node descriptors to render a graph. Invalid parameter values, missing references, channel mismatches and cycles remain editable and are reported through Core preflight.

Python export uses actual `ModelBuilder.from_config` or `GraphModelBuilder.from_config` calls. JSON is embedded as a Python string literal and parsed with `json.loads`. Disabled drafts contain an explicit execution guard. Python snippets are downloads for the user's Python environment. The HTTP service accepts JSON configurations.

## Diagnostics and resources

Automatic preflight is debounced by 300 ms. A new edit aborts the previous fetch and invalidates its sequence number. Explicit validation remains available when PROJECT auto-preview is disabled. Node shapes remain unknown until Core returns them.

Frontend JSON input rejects duplicate keys, nonfinite numbers, excessive nesting and oversized containers before updating a configuration. Uploads use strict UTF-8 decoding. API serialization also rejects nonfinite values, preventing their conversion into JSON null.

Integer literals that cannot be represented exactly as JavaScript Number use BigInt internally. Input fields display decimal text and the custom JSON encoder emits the original exact integer as a numeric token. It sends no decimal-string substitution to Core. Structured history, saved drafts, responses, Python export and JSON export retain the value. Core rejects values outside its actual bounds while the invalid draft remains editable. Native result uploads retain the original numeric text independently.

Core limits remain 64 graph nodes, 256 edges, fan-in 8, fan-out 16, eight inputs and eight outputs, dimensions up to 8192 and the 10M conservative parameter budget. Attention retains 1024-token and 4194304-matrix-element bounds. WeightedSum retains raw fixed or learnable weights.

SCC diagnostics expose sorted `cycles` and `blocked_nodes`. Each cycle diagnostic's `layer_id` belongs to its actual SCC. Node metadata includes `cycle_member` and `blocked_by_cycle`. Acyclic ordering remains deterministic.

Studio build jobs use the existing Core parameter budget. The separate short forward/backward, training and benchmark profiles admit up to 1M estimated activation elements. Training and benchmark admission uses the actual supplied batch count. Core preview and exported Core configurations retain the existing Core limits.

The short runtime profile also requires a Core parameter upper bound at most 1M. The separate explicit exact-count build uses the same 1M parameter bound and does no forward pass. Ordinary Core build and exported configurations keep their existing budgets. Node cards display a brief component purpose, ports and status; dimensions remain opt-in.

## Türkçe düzenleme rehberi

Paletten bileşen ekleyin veya canvas'a sürükleyin. Düğümü seçerek kalıcı kimliğini, sıralı giriş referanslarını ve tüm Registry parametrelerini düzenleyin. Çıkış portuna, ardından hedef giriş portuna tıklayın. Tek girişli bileşende referans değiştirilir. Fusion düğümünde sıralı listeye eklenir. Listeden bağlantı kaldırma ve sıralama açıkça yapılır.

Branch, external skip ve birleştirmeler normal düğüm/bağlantılarla ifade edilir. Kanal eşleştirmede `pointwise_conv2d`, uzamsal eşleştirmede `resize` gibi açık düğümler kullanın. Add/Multiply eş şekil, Concat seçili eksen dışı eş boyut, WeightedSum girdi başına ham ağırlık ister. Adlandırılmış çıkışı yeni düğüme bağlayın. Bağımsız, çıktıya ulaşmayan düğüm Core tarafından geçersiz sayılır.

Düğüm kartında ad, bileşen ve tanı durumu gösterilir. Boyutları göster seçeneği şekil ve sayıları açar. Ham tensor/metadata yalnız teknik görünümde bulunur. Inspector kapatılabilir; mobilde alt paneldir. Parametre değişikliği 300 ms sonra Core ön doğrulamasını yeniler. Eski istek iptal edilir ve eski yanıt yeni taslağa uygulanmaz. Geçersiz değer sessizce düzeltilmez.

Yeni düğüm ve içe aktarılan graph görünümü sığdırılır. Arka planı sürükleyerek kaydırın, tekerlek veya düğmelerle yakınlaştırın. Düğümü sürükleyin ya da odaklayıp ok tuşlarıyla taşıyın. Geri al/yinele 80 durum saklar. Yeniden adlandırma referansları günceller. Silme sonrasında referanslar açık onarım için korunur. Düğümü kapatmak bağlantıları korur ve yürütmeyi engeller.

Taslağı kaydetmek konumları ve düğüm durumunu SQLite'a yazar. Revision çakışması eski taslağın üstüne yazmayı reddeder. Core JSON, Studio taslak paketi ve gerçek Python API karşılığı indirilebilir. Büyük tam sayılar decimal metin olarak düzenlenir, BigInt ile kayıpsız sayısal JSON olarak taşınır. Schema1/schema2 JSON yolu, schema3 görsel düzenleme yolu desteklenir.
