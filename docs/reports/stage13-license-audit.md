# Stage 13 source-license audit

Reviewed on 2026-11-12 against HYSo HEAD `3c0bd5eca5953163d4e00760add41e8ffdc0592e`.
The user's commercial-use policy takes precedence over completing the initial eight-family target.

## Decisions

Seven model families and ten blocks have identified source-code permissions compatible with commercial use subject to their license conditions. This finding covers source copyright permissions. Commercial deployment clearance, patent freedom to operate, trademark permission and trained-model rights remain unverified.

FasterNet T0/T1/T2 and PartialConv are excluded. The reviewed `models/fasternet.py` starts with a Microsoft copyright and MIT reference. The pinned repository tree contains no LICENSE, NOTICE or acknowledgement file, and the three YAML configurations have no explicit license header. This leaves the extent of the grant and configuration provenance unresolved under the user's policy. The earlier HYSo implementation was removed from runtime code, exports, Registry, Builder configuration whitelist and examples. Review copies remain outside the checkout in `%LOCALAPPDATA%/hyso-stage13-20261112/excluded-review/`. Existing M3 and custom CNN behavior is preserved.

A future independently implemented partial-channel spatial operator is technically feasible from the published mathematical description. The current session already consulted the upstream implementation. An independent-source process and its legal sufficiency were not established, so no replacement is represented as cleared.

## Source inventory

| Source | Audited revision | License and scope | HYSo use |
|---|---|---|---|
| pytorch/vision | `3ac97aa9120137381ed1060f37237e44485ac2aa`, tag v0.20.1 | BSD-3-Clause, root LICENSE | Runtime wrappers, standalone adapters, pure metadata and channel rounding |
| pytorch/pytorch | `a8d6afb511a69687bbb2b7e88a3cf67917e1697e`, tag v2.5.1 | Root BSD-style license and NOTICE with named holders | Existing runtime dependency, tensor operators, normalization, activations, Conv–BN fusion |
| huawei-noah/Efficient-AI-Backbones | `f90e129b645c3b1684fe07cd361cd557d0ad71f7` | `ghostnet_pytorch/License.txt`, Apache-2.0, January 2004, and directory-specific third-party notice | GhostNetV1 configuration and operator topology adaptations |
| DingXiaoH/RepVGG | `eae7c5204001eaf195bbe2ee72fb6a37855cce33` | MIT, Copyright 2020 DingXiaoH | A0/A1 topology and reparameterization reference |
| apple/ml-mobileone | `3cebdf04a1524c447ab5e44a5235524f8da46f8d` | Apple copyright license in LICENSE, AML-style terms, Copyright 2022 Apple, plus RepVGG MIT in ACKNOWLEDGEMENTS | s0/s1/s2 topology and conversion reference |
| JierunChen/FasterNet | `25b5dc75d047d61878dc59061d31ef4608507950` | File-level MIT reference, missing repository license text and configuration grant | Excluded, including the consulted PartialConv implementation |

Repository and immutable document links, complete retained notices and SHA256 hashes are provided in [third-party notices](../../THIRD_PARTY_NOTICES.md). Sources were obtained from the official repositories. FasterNet's file header can be inspected in its [pinned source](https://github.com/JierunChen/FasterNet/blob/25b5dc75d047d61878dc59061d31ef4608507950/models/fasternet.py).

## Every family and block

| Component | Upstream code or topology | HYSo files | Decision |
|---|---|---|---|
| MobileNetV2 | torchvision `models/mobilenetv2.py` | `mobile_architectures.py`, `m4_config.py`, `m4_registry.py` | BSD-3-Clause notices retained |
| MobileNetV3 Small/Large | torchvision `models/mobilenetv3.py` | Same three files | BSD-3-Clause notices retained |
| EfficientNetV2 S/M/L | torchvision `models/efficientnet.py` | Same three files | BSD-3-Clause notices retained |
| ShuffleNetV2 four widths | torchvision `models/shufflenetv2.py` | Same three files | BSD-3-Clause notices retained |
| GhostNetV1 | Huawei `ghostnet_pytorch/ghostnet.py` | `mobile_architectures.py`, `m4_blocks.py`, `m4_config.py`, `m4_registry.py` | Apache-2.0 and original third-party notice retained |
| FasterNet T0/T1/T2 | FasterNet `models/fasternet.py`, `cfg/fasternet_t{0,1,2}.yaml` | Earlier additions withdrawn from M4 files | Excluded |
| RepVGG A0/A1 | RepVGG `repvgg.py` | `m4_reparam.py`, `reparam_architectures.py`, `m4_config.py`, `m4_registry.py` | MIT notice retained |
| MobileOne s0/s1/s2 | Apple `mobileone.py` and RepVGG acknowledgement | Same four files | Apple license and MIT acknowledgement retained |
| InvertedResidual | torchvision MobileNetV2 adapter | `m4_blocks.py`, `m4_registry.py` | BSD-3-Clause |
| MBConv, FusedMBConv | torchvision EfficientNet adapters | Same two files | BSD-3-Clause |
| GhostConv, GhostBottleneck | Huawei GhostNetV1 topology, torchvision convolution/SE operators | Same two files | Apache-2.0 and BSD-3-Clause |
| ChannelShuffle | HYSo reshape/transpose implementation, torchvision used as reference in tests | Same two files | HYSo implementation, dependency notices retained |
| ShuffleUnit | torchvision ShuffleNetV2 adapter | Same two files | BSD-3-Clause |
| PartialConv | Consulted FasterNet spatial operator | Earlier addition withdrawn | Excluded |
| RepVGGBlock | RepVGG topology, PyTorch fusion helper | `m4_reparam.py`, `m4_registry.py` | MIT and PyTorch notices |
| MobileOneBlock | Apple topology, shared HYSo fusion, existing HYSo SEBlock | Same two files | Apple and RepVGG MIT notices |
| DropBlock2D | HYSo implementation of valid-origin masks, pooling and normalization | `m4_blocks.py`, `m4_registry.py` | HYSo implementation, existing PyTorch dependency |

All HYSo filenames in this table are relative to `hyso/models/`. Registry and Builder integration and tests are HYSo glue. Original upstream Python files are retained only as audit references outside the repository. Conservative attribution is retained for topology adaptations even where the HYSo code uses different module composition and state keys.

## License conditions and file exceptions

- BSD-3-Clause permits source/binary redistribution with modifications and imposes copyright, condition and disclaimer retention. Binary redistribution must reproduce these notices in its accompanying materials. Its non-endorsement clause applies. The reviewed torchvision model, `_utils.py`, `ops/misc.py` and `ops/stochastic_depth.py` files contain no separate conflicting license header. The repository's TensorFlow license under damaged-JPEG test assets belongs to unused test data.
- The GhostNet directory has its own Apache-2.0 license and third-party notice. A repository-wide license assumption would have missed these files. Both are retained in full. The source identifies Huawei changes dated 2020-06-09 and earlier MobileNetV3/timm origins. The upstream notice retains Ross Wightman and Duo Li material, alongside additional upstream acknowledgements. These notices are preserved conservatively. Tensorpack, TensorFlow, deit, regnet and timm are not added as HYSo dependencies.
- Apache-2.0 requires a license copy, preservation of applicable notices, and prominent modification notices. HYSo's adapted Ghost source/configuration files identify the HYSo changes. Contributor patent permissions in section 3 have a defined scope and termination conditions. Section 6 limits trademark permissions.
- RepVGG MIT permits use, modification, distribution and sale subject to retention of its copyright and permission text. Its source header points to the root MIT license, which was verified at the pinned revision.
- Apple's LICENSE grants use, reproduction, modification and source/binary redistribution. The reviewed text has no noncommercial restriction. It also expressly reserves other rights including patent rights, limits use of Apple names and marks for endorsement, and directs readers to ACKNOWLEDGEMENTS. The latter contains RepVGG MIT terms. Both complete documents are preserved. The SPDX AML description corroborates the license family, while the exact upstream text governs this adaptation. This source is not relabeled as standard MIT. [Apple license](https://github.com/apple/ml-mobileone/blob/3cebdf04a1524c447ab5e44a5235524f8da46f8d/LICENSE), [acknowledgements](https://github.com/apple/ml-mobileone/blob/3cebdf04a1524c447ab5e44a5235524f8da46f8d/ACKNOWLEDGEMENTS), [SPDX AML](https://spdx.org/licenses/AML.html).
- PyTorch's root license names several holders and refers to accompanying subcomponent terms. Its LICENSE and NOTICE are retained. HYSo continues to depend on an installed PyTorch distribution. Redistribution of PyTorch/torchvision binaries requires the full notices shipped with those distributions, including compiled/transitive components. This focused review does not certify their complete dependency trees.

## Rights evaluated separately

| Area | Finding |
|---|---|
| Architecture ideas | Papers and configuration descriptions establish technical references. Source copyright permission and implementation provenance are recorded separately. Reading a paper provides no universal patent clearance. |
| Source code | Identified permissive grants and exact notices are retained for included material. FasterNet/PartialConv have unresolved provenance/licensing scope and are excluded. |
| Weights and datasets | No pretrained weights or datasets were downloaded or distributed. Random initialized smoke weights are synthetic local artifacts. Rights for future pretrained or user-trained artifacts require their own evidence. |
| Trademarks | Architecture names identify compatibility and topology. No Apple, Huawei, PyTorch or author endorsement is claimed. Logo and trademark permissions were not obtained. |
| Patents | No patent landscape or claim analysis was performed. MIT/BSD copyright permissions supply no universal patent assurance. Apache section 3 covers defined contributor claims. Apple's text expressly excludes implied additional patent rights. Freedom to operate remains NOT_EVALUATED. |

## Distribution and remaining boundaries

`THIRD_PARTY_NOTICES.md` is included in sdist through MANIFEST.in and in wheel license metadata through `setup.cfg` license_files. Packaging tests compare its complete bytes against the checkout. License text is retained verbatim, including its original punctuation. The HYSo MIT metadata continues to describe HYSo-owned code, and the bundled notices define the applicable third-party terms.

The current installed versions are torch `2.5.1+cu121` and torchvision `0.20.1+cu121`. The review checks upstream release sources and the installed torchvision license. Other versions permitted by the existing dependency ranges require source/license re-evaluation when changed. No new dependency is introduced.

This audit records a source-code licensing decision for this task. It supplies no commercial-use certification or guarantee that all legal risks have been eliminated. Ownership disputes, jurisdiction-specific issues, patents, trademarks and future weight/data licenses remain outside the verified evidence. M3, custom CNN implementation, native motors, acceptance evidence, checkpoints and the 47 protected files are covered by the separate final integrity audit.

## Pinned source evidence

The complete source bytes and machine-readable manifest remain outside the checkout under `%LOCALAPPDATA%/hyso-stage13-20261112/licenses/`. SHA256 values identify the reviewed files.

| Repository file | SHA256 |
|---|---|
| [pytorch/vision/torchvision/models/mobilenetv2.py](https://raw.githubusercontent.com/pytorch/vision/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/mobilenetv2.py) | `bfd711040a7bfc2ff9d091648c667496ebee878e680988189f7ee9706d942fca` |
| [pytorch/vision/torchvision/models/mobilenetv3.py](https://raw.githubusercontent.com/pytorch/vision/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/mobilenetv3.py) | `f97937da6fd6767f9aa7cc022f84e5ef08e244e2300f085ab089d84ccc303ab1` |
| [pytorch/vision/torchvision/models/efficientnet.py](https://raw.githubusercontent.com/pytorch/vision/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/efficientnet.py) | `e2ac9ea17918185c94b010edf13ca00d8c9c30cb75ce11b007f9783e6a0fbf88` |
| [pytorch/vision/torchvision/models/shufflenetv2.py](https://raw.githubusercontent.com/pytorch/vision/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/shufflenetv2.py) | `f381623df9212694b0e90f4999aba0e4c5b9aa65823b756100f17ad4488c9fb4` |
| [pytorch/vision/torchvision/models/_utils.py](https://raw.githubusercontent.com/pytorch/vision/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/models/_utils.py) | `4bcb830fb99a35e7f2f9f116ea6a73f1d154ebc69c2b51f137492dd6aa64903a` |
| [pytorch/vision/torchvision/ops/misc.py](https://raw.githubusercontent.com/pytorch/vision/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/ops/misc.py) | `c859caec64fd38231f0eb9f836d41729d8796eba22d71a1c2fde12a32aa15aec` |
| [pytorch/vision/torchvision/ops/stochastic_depth.py](https://raw.githubusercontent.com/pytorch/vision/3ac97aa9120137381ed1060f37237e44485ac2aa/torchvision/ops/stochastic_depth.py) | `21267d9e827264bc694c6fb06b65663ecebaaa3855b0fed9c561efba65923f75` |
| [pytorch/pytorch/torch/nn/utils/fusion.py](https://raw.githubusercontent.com/pytorch/pytorch/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/torch/nn/utils/fusion.py) | `f2de1ca605a4df871a5250f04c6f45296ef4a9944e7850055f71f62b4ca5a892` |
| [huawei-noah/Efficient-AI-Backbones/ghostnet_pytorch/ghostnet.py](https://raw.githubusercontent.com/huawei-noah/Efficient-AI-Backbones/f90e129b645c3b1684fe07cd361cd557d0ad71f7/ghostnet_pytorch/ghostnet.py) | `cfd6fe7e4705406b68111680dc5ab6904512f9f4a314c6f29052d94b0a74be5b` |
| [JierunChen/FasterNet/models/fasternet.py](https://raw.githubusercontent.com/JierunChen/FasterNet/25b5dc75d047d61878dc59061d31ef4608507950/models/fasternet.py) | `497debd1bc63a4c467b52af58d671dd9831287c193ff4922bb41cb8826ddeeb9` |
| [JierunChen/FasterNet/cfg/fasternet_t0.yaml](https://raw.githubusercontent.com/JierunChen/FasterNet/25b5dc75d047d61878dc59061d31ef4608507950/cfg/fasternet_t0.yaml) | `73f6f79c7322f22d0afb07cfed19260fbc1cdd5f4d41b8d3b3bff54ee4c080d1` |
| [JierunChen/FasterNet/cfg/fasternet_t1.yaml](https://raw.githubusercontent.com/JierunChen/FasterNet/25b5dc75d047d61878dc59061d31ef4608507950/cfg/fasternet_t1.yaml) | `f4424524cd79c8f9e414b693ee525e374549d38e61bf3fb91f9281cccfba938d` |
| [JierunChen/FasterNet/cfg/fasternet_t2.yaml](https://raw.githubusercontent.com/JierunChen/FasterNet/25b5dc75d047d61878dc59061d31ef4608507950/cfg/fasternet_t2.yaml) | `343de9542700570745a370daa25b7ef00559762bb5f7530e99c039b52fcdcc68` |
| [DingXiaoH/RepVGG/repvgg.py](https://raw.githubusercontent.com/DingXiaoH/RepVGG/eae7c5204001eaf195bbe2ee72fb6a37855cce33/repvgg.py) | `8d8c9698a97b5a8e497a0e6072f3235c24385a73c7d1cf3e42bf99a1b67b47c9` |
| [apple/ml-mobileone/mobileone.py](https://raw.githubusercontent.com/apple/ml-mobileone/3cebdf04a1524c447ab5e44a5235524f8da46f8d/mobileone.py) | `70b58756631dcf4ed2e5513fdc2359579421d5f342d5e0dcd89114ce823f3a80` |
