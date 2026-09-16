# 视频来源、原始文件与派生文件 provenance 规范

## 目的

这份文档是 VideoSeek 视频资产的唯一 provenance 约定。它解决三个问题：

1. 以后能确认某个 MP4 究竟来自哪个公开数据集和官方 ZIP 成员；
2. 能确认轨迹使用的是原视频还是 640×360 派生视频；
3. 在任何清理动作前，能证明文件可以按原来源或归档恢复，而不是只留下一个无法复核的文件名。

记录格式由 [video_provenance_v0_1.json](../src/tracing/schema/video_provenance_v0_1.json) 约束。每个 JSONL 记录对应一个逻辑 `video_id`，原始文件放在 `original`，派生文件放在 `derivatives` 数组中。

## 当前决策

- 原始视频是 source-of-truth，当前继续保留，不能被 640×360 文件覆盖。
- 640×360 是独立派生 variant；它必须记录 `parent_sha256 = original.sha256`，并使用独立 collection/variant 标识。
- 用原视频生成且通过 validator 的旧轨迹仍可直接做 trace-level predictor、workload 和调度实验；新增派生视频产生的轨迹不能和原视频轨迹混成同一运行条件。
- 本轮只落记录和执行计划，不删除视频、不重跑轨迹。任何删除都要等 R5 provenance/deletion gate 通过并得到单独确认。
- 2026-08-17 例外：用户明确授权为释放远端空间删除旧 `P_dev` 的 300 个原始 MP4；本次不保留本地归档，但保留 provenance、SHA-256、官方 archive 定位和来源 URL。该例外不适用于 holdout 或 scheduler 视频。

## 字段说明

| 位置 | 字段 | 作用 | 填写规则 |
| --- | --- | --- | --- |
| 根 | `collection_id`、`video_id` | 标识数据 collection 和逻辑视频 | v1/v2 不混用；ID 不因转码改变 |
| `source` | `dataset`、`dataset_revision`、`source_url` | 记录公开来源 | 不凭文件名猜来源；未知 revision 写 `null` |
| `source.official_archive` | `repository`、`shard`、`member` | 定位官方 ZIP | Video-MME 使用官方仓库、shard 编号和 ZIP member 名称 |
| `source.official_archive` | `shard_size_bytes`、`compressed_size_bytes`、`uncompressed_size_bytes` | 复查下载是否完整 | 直接使用官方索引/下载报告 |
| `source.official_archive` | `crc32`、`local_offset` | 快速校验和按范围重新提取 | `crc32` 使用十进制整数；不手填估计值 |
| `original` | `path`、`sha256`、`size_bytes` | 识别原始字节内容 | `path` 用相对项目根的逻辑路径；SHA-256 小写 64 位十六进制 |
| `original` | `width`、`height`、`duration_s` | 固定视觉和时间条件 | 由 `ffprobe` 读取，不能从文件名推断 |
| `derivatives[]` | `variant_id`、`path` | 区分 640×360 等输入 | 例如 `derived_640x360_crf23` |
| `derivatives[]` | `parent_sha256` | 把派生文件绑定到具体原视频 | 必须等于同一记录的 `original.sha256` |
| `derivatives[]` | `sha256`、`size_bytes`、尺寸、时长 | 复现实验输入和容量计算 | 派生后重新计算，不能复用原文件值 |
| `derivatives[]` | `encoder` | 记录转码条件 | 包含实际使用的编码器、preset/CRF 等关键参数 |
| `retrieval` | `retrieved_at`、`verified_at`、`method` | 记录获取和校验时间 | 采用 ISO 8601；历史获取时间未知时 `retrieved_at=null`，不能用回填时间冒充；`method` 写真实命令/方式 |
| `retention` | `original_status`、`archive_path`、`archive_sha256` | 说明原始文件是否仍在及归档位置 | 当前文件应为 `present`；没有归档就写 `null` |
| `retention` | `archive_restore_tested`、`deletion_eligible` | 形成删除闸门 | 默认 `false`；所有条件满足前 `deletion_eligible=false` |

## R5 执行顺序

### R5-A：冻结来源和 split

以 `videomme_600_source_v2.jsonl` 为选择输入，先按视频去重并冻结视频级 split。该清单目前只有 `video_id/source_url/video_path` 等选择字段，不能直接作为完整 provenance 清单。

### R5-B：回填原始 provenance

为 v1 和 v2 分别生成以下 collection 清单：

- `data/manifests/video_provenance_v1_legacy_core.jsonl`
- `data/manifests/video_provenance_v2_expanded.jsonl`

每一行都必须补齐 schema 的 `source`、`original`、`retrieval` 和 `retention`。官方 Video-MME 下载报告可以复用其 `shard`、`member/name`、CRC32、大小、offset 和 SHA-256；尺寸和时长仍需以实际文件的 `ffprobe` 结果为准。

### R5-C：生成可选派生 variant

如果容量 gate 选择 640×360，派生文件写入独立目录/collection，并在同一 provenance 记录中追加 derivative。先做抽样的视觉/轨迹一致性检查，再冻结编码参数；不能因为 pilot 的平均节省率就直接删除原文件。

### R5-D：删除门槛

只有同时满足以下条件，某个原始文件才可进入候选删除状态：

1. 原始 SHA-256、字节数、分辨率、时长和官方 ZIP 定位字段齐全；
2. 每个派生文件的 `parent_sha256` 与原始 SHA-256 相等，派生文件自身 SHA-256 已核验；
3. 原始文件有可恢复归档，归档 SHA-256 已记录，且至少做过一次实际恢复测试；
4. 相关 trace/workload/experiment manifest 已引用 `video_id + variant_id + sha256`，不会退化成只依赖路径；
5. C2 测试所需的全新原视频仍保留或可恢复；
6. 项目控制面和清单更新后，得到针对目标文件的明确删除确认。

在门槛通过前，`deletion_eligible` 必须为 `false`。即使某个文件已经生成 640×360 版本，也不能自动删除原视频。

本项目 2026-08-17 对旧 `P_dev` 做了范围限定的用户授权例外：归档恢复测试被明确豁免，目标记录在删除后标记为 `deleted_after_gate`；这不是其他视频的默认删除规则。

## 验收标准

- provenance schema JSON 可解析，新增记录通过该 schema；
- v1/v2 的记录数按唯一 `video_id` 统计，并能和 split/source manifest 对账；
- 任一派生文件都能沿 `parent_sha256` 找回唯一原视频；
- 原视频删除前能从官方 ZIP 或归档恢复并重新得到同一 SHA-256；
- 本地和远端只同步控制面、schema、清单和文档；视频/模型/原始 trace 不进入 Git。

## 当前状态与下一步

现有官方下载报告已具备完整 archive/校验字段。当前回填结果为：v1 legacy `64/64` 条有效、官方 archive `64/64` 覆盖；v2 expanded `600/600` 条唯一记录有效、官方 archive `600/600` 覆盖，6 条派生记录已补 `parent_sha256`。其中旧 `P_dev` 300 条已按 2026-08-17 用户授权删除原始 MP4，保留来源 URL 和完整 provenance；其余 300 个 holdout/scheduler 视频仍在远端。删除审计见 `results/processed/video_delete_pdev_20260817/`。远端不能稳定访问 `huggingface.co:443`，因此未来恢复采用“按来源 URL/archive 定位 → 本地获取/校验 → rsync 到远端”的路径。
