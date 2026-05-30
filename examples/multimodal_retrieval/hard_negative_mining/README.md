# Multimodal Hard Negative Mining Examples

本目录说明如何在 Nexus 里给多模态 retrieval 训练数据挖掘 hard negatives，并把挖好的 JSONL 直接接入 `Nexus.training.embedder.multimodal_retrieval` 训练。

相比随机负例，hard negative 更能逼模型学会细粒度区分。

## 入口

在 Nexus 仓库根目录运行：

```bash
python -m Nexus.data_mining.multimodal_hard_negative --help
```

代码位置：

```text
Nexus/data_mining/multimodal_hard_negative/
```

输出仍然是 Nexus 原生训练格式：`query`、`pos`、`neg`，可选 `pos_scores`、`neg_scores`，所以可以直接作为训练数据。

## 支持模式

| 模式 | 作用 | 什么时候用 |
| --- | --- | --- |
| `mock` | 不加载模型，用确定性伪分数跑通 JSONL 读写和报告输出。 | 第一次检查数据格式、路径、输出目录。 |
| `teacher_only` | 只用强 teacher embedding 模型给候选打分，选最像 query 的非正例。 | 第一阶段可选方案。 |
| `student_aware` | 同时用 teacher 和 student；优先挑学生容易错排、teacher 认为仍合理的负例。 | 已经训出一个 adapter 后，做第二阶段继续精修。|

## 输入格式

输入是 Nexus multimodal retrieval 的训练 JSONL，每行一条训练样本：

```json
{
  "query": {"text": "Find the image of a red dress."},
  "pos": [
    {"text": "A woman wearing a red dress.", "image_path": "images/red_dress.jpg"}
  ],
  "neg": [
    {"text": "A woman wearing a blue dress.", "image_path": "images/blue_dress.jpg"}
  ]
}
```

挖掘后会生成类似：

```json
{
  "query": {"text": "Find the image of a red dress."},
  "pos": [
    {"text": "A woman wearing a red dress.", "image_path": "images/red_dress.jpg"}
  ],
  "neg": [
    {"text": "A red skirt on a model.", "image_path": "images/red_skirt.jpg"},
    {"text": "A red coat displayed in a shop.", "image_path": "images/red_coat.jpg"}
  ],
  "pos_scores": [0.83],
  "neg_scores": [0.74, 0.69]
}
```

## 最小 mock 检查

这个命令不需要 GPU，也不需要模型，只验证代码和 JSONL 格式。

```bash
bash examples/multimodal_retrieval/hard_negative_mining/run_mock_smoke.sh
```

如果默认 `python` 不是你的 Nexus 环境，显式指定当前环境即可：

```bash
PYTHON=/path/to/your/python \
  bash examples/multimodal_retrieval/hard_negative_mining/run_mock_smoke.sh
```

也可以用 mock 快速验证某个候选源参数是否能跑通，例如测试“只加入本行自己原始负例”的模式：

```bash
CANDIDATE_SOURCE=all_pos_plus_own_neg \
PYTHON=/path/to/your/python \
  bash examples/multimodal_retrieval/hard_negative_mining/run_mock_smoke.sh
```

默认输出：

```text
outputs/hard_negative_mining/mock_smoke/train.mock_hardneg.jsonl
outputs/hard_negative_mining/mock_smoke/report.json
outputs/hard_negative_mining/mock_smoke/manifest.json
outputs/hard_negative_mining/mock_smoke/audit.jsonl
```

如果这一步都失败，优先修数据格式，不要直接上大模型挖掘。

## Teacher-only 挖掘

推荐第一阶段先用强 teacher 模型挖。例如使用 `Qwen3-VL-Embedding-8B`：

```bash
export TEACHER_MODEL=/path/to/Qwen3-VL-Embedding-8B
export INPUT_TRAIN_JSONL=/path/to/train.jsonl
export OUTPUT_DIR=/path/to/output/hard_negative_mining/my_teacher_run
export DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
export BATCH_SIZE=16
export CANDIDATE_SOURCE=all_pos
export SCORE_BLOCK_SIZE=4096
bash examples/multimodal_retrieval/hard_negative_mining/run_teacher_only.sh
```

主要输出：

```text
$OUTPUT_DIR/train.teacher_hardneg.jsonl
$OUTPUT_DIR/report.json
$OUTPUT_DIR/manifest.json
$OUTPUT_DIR/audit.jsonl
```

我们在 MMEB v2 retrieval 实验中已经验证过 teacher-only 路线：`EDIS` 从普通训练 `hit@1=95.40` 提升到 hard negative 训练 `hit@1=96.90`，超过官方 `Qwen3-VL-Embedding-8B` 的 `96.30`。

如果数据集本身每条样本就带了比较可信的原始 `neg`，可以测试更保守的 own-neg 模式：

```bash
export CANDIDATE_SOURCE=all_pos_plus_own_neg
bash examples/multimodal_retrieval/hard_negative_mining/run_teacher_only.sh
```

## 候选池怎么扩展

Hard negative mining 的核心问题是：query 要和哪些候选文档做相似度排序。候选越贴近评测，挖出来的负例通常越有价值。

现在支持三类候选池来源，可以叠加使用：

| 来源 | 参数 | 解释 |
| --- | --- | --- |
| 训练集自身正例 | `--candidate_source all_pos` | 把训练集中所有 `pos` 当作候选池，这是最稳的默认方案。 |
| 训练集所有 doc | `--candidate_source train_all_docs` | 把训练集中所有 `pos` 和所有原始 `neg` 都放进候选池。`train_all_docs` 等价于 `all_pos_and_existing_neg`，名字更直观。 |
| 外部候选 JSONL | `--candidate_corpus_jsonl /path/a.jsonl,/path/b.jsonl` | 额外加入 corpus 文件，比如 M-BEIR 官方 candidate pool。 |
| MMEB v2 评测 corpus | `--mmeb_eval_corpus_datasets EDIS,WebQA` | 直接复用 MMEB v2 评测 parser，读取指定评测集的 candidate/corpus。 |

只用训练集所有 doc：

```bash
python -m Nexus.data_mining.multimodal_hard_negative \
  --mode mock \
  --input_train_jsonl examples/multimodal_retrieval/hard_negative_mining/data/train_tiny.jsonl \
  --output_train_jsonl outputs/hard_negative_mining/mock_train_all_docs/train.jsonl \
  --candidate_source train_all_docs \
  --top_k 5 \
  --num_hard_negatives 2 \
  --no-require_existing_media
```

训练集所有 doc + M-BEIR EDIS 外部 candidate pool：

```bash
python -m Nexus.data_mining.multimodal_hard_negative \
  --mode teacher_only \
  --teacher_model /path/to/Qwen3-VL-Embedding-8B \
  --input_train_jsonl /path/to/edis/train.jsonl \
  --output_train_jsonl /path/to/output/edis_train_docs_plus_corpus/train.teacher_hardneg.jsonl \
  --candidate_source train_all_docs \
  --candidate_corpus_jsonl /path/to/mbeir_edis_task2_cand_pool.jsonl \
  --top_k 200 \
  --num_hard_negatives 3 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --batch_size 16 \
  --score_block_size 4096 \
  --filter_duplicate_media \
  --write_scores
```

训练集所有 doc + MMEB v2 评测 corpus by dataset name：

```bash
python -m Nexus.data_mining.multimodal_hard_negative \
  --mode teacher_only \
  --teacher_model /path/to/Qwen3-VL-Embedding-8B \
  --input_train_jsonl /path/to/edis/train.jsonl \
  --output_train_jsonl /path/to/output/edis_train_docs_plus_mmeb_eval_corpus/train.teacher_hardneg.jsonl \
  --candidate_source train_all_docs \
  --mmeb_eval_corpus_datasets EDIS \
  --mmeb_data_basedir /path/to/MMEB/vlm2vec_eval \
  --mmeb_hf_cache_dir /path/to/huggingface/hub \
  --top_k 200 \
  --num_hard_negatives 3 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --batch_size 16 \
  --score_block_size 4096 \
  --filter_duplicate_media \
  --write_scores
```

可以一次指定多个评测集：

```bash
--mmeb_eval_corpus_datasets EDIS,WebQA
```

如果默认配置文件里找不到某个评测集，可以显式指定 MMEB v2 YAML：

```bash
--mmeb_eval_dataset_config examples/multimodal_retrieval/evaluation/mmeb_v2/configs/image_retrieval.yaml,examples/multimodal_retrieval/evaluation/mmeb_v2/configs/visdoc_retrieval.yaml
```

说明：`--mmeb_eval_corpus_datasets` 会复用 MMEB v2 评测 parser。有些 image retrieval parser 需要从 Hugging Face 读取 query/candidate 元数据；离线环境必须命中本地 Hugging Face cache。代码会自动尝试 `HF_HUB_CACHE`、`HUGGINGFACE_HUB_CACHE`、`OFFLINE_HF_HUB_ROOT`、`HF_HOME/hub` 和 `~/.cache/huggingface/hub`，也可以用 `--mmeb_hf_cache_dir` 显式指定。

挖完后，`report.json` 会新增这些字段：

- `candidate_count_before_extra`：只用训练 JSONL 构建出的候选数量。
- `candidate_count_after_extra`：追加外部 corpus 和 MMEB eval corpus 后的候选数量。
- `extra_candidate_sources`：每个额外来源原始读入多少、去重后新增多少。

## Student-aware 挖掘

当你已经有一个第一阶段 adapter 后，可以用它作为 student 继续挖：

```bash
export TEACHER_MODEL=/path/to/Qwen3-VL-Embedding-8B
export STUDENT_MODEL=/path/to/student_adapter_or_model
export INPUT_TRAIN_JSONL=/path/to/train.jsonl
export OUTPUT_DIR=/path/to/output/hard_negative_mining/my_student_aware_run
export DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
export BATCH_SIZE=8
export CANDIDATE_SOURCE=all_pos
export SCORE_BLOCK_SIZE=4096
bash examples/multimodal_retrieval/hard_negative_mining/run_student_aware.sh
```

我们在 `EDIS` 上验证过第二阶段：从 teacher-only adapter 继续 student-aware 后，`hit@1` 从 `96.90` 到 `97.00`。

## 挖完以后怎么训练

把训练配置里的 `train_data` 指向挖好的 JSONL：

```json
{
  "train_data": ["/path/to/train.teacher_hardneg.jsonl"],
  "train_group_size": 4,
  "query_instruction_for_retrieval": "Find an image that matches the given query.",
  "passage_instruction_for_retrieval": null
}
```

然后正常跑 Nexus multimodal retrieval 训练：

```bash
torchrun --nproc_per_node=8 -m Nexus.training.embedder.multimodal_retrieval \
  --model_config /path/to/model_config.json \
  --data_config /path/to/data_config.json \
  --training_config /path/to/training_config.json
```

## 常用参数解释

| 参数 | 解释 |
| --- | --- |
| `--input_train_jsonl` | 原始 Nexus 训练 JSONL。 |
| `--output_train_jsonl` | 挖好 hard negatives 后的新 JSONL。 |
| `--candidate_train_jsonl` | 可选，单独指定候选池 JSONL；默认用输入训练集自身。 |
| `--candidate_source` | 候选来源。`all_pos` 只用所有正例；`all_pos_and_existing_neg` 把所有行原始负例全局加入；`train_all_docs` 是同义但更直观的名字；`all_pos_plus_own_neg` 只让每行自己的原始负例参与本行竞争。 |
| `--candidate_corpus_jsonl` | 额外候选 corpus JSONL，可逗号分隔多个文件。支持 Nexus train-style、MMEB eval-style `cand_input`、M-BEIR `txt/img_path` 等常见格式。 |
| `--mmeb_eval_corpus_datasets` | 额外读取 MMEB v2 评测集 corpus/candidates，可逗号分隔多个评测集名。 |
| `--mmeb_eval_dataset_config` | 可选，指定 MMEB v2 评测 YAML；不写时会搜索 examples 里的常见 image/video/visdoc retrieval 配置。 |
| `--mmeb_data_basedir` | MMEB v2 评测数据根目录，含义和评测脚本里的 `data_basedir` 一致。 |
| `--mmeb_generated_media_basedir` | VisDoc 等需要生成页面图时的缓存目录。 |
| `--mmeb_hf_cache_dir` | MMEB v2 元数据集的 Hugging Face hub cache 根目录。不指定时会尝试常见 Hugging Face cache 环境变量和 `~/.cache/huggingface/hub`。 |
| `--mmeb_num_sample_per_subset` | 只建议 smoke test 使用，用来少量抽样评测 corpus；正式挖掘不要设置。 |
| `--top_k` | 每个 query 先看 teacher 排名前多少的候选。 |
| `--num_hard_negatives` | 每条训练样本最终保留几个 hard negatives。 |
| `--score_block_size` | 大候选池分块 exact top-k 的块大小，默认 `4096`。它不截断候选，只避免一次性构造完整 `query x candidate` 大矩阵。 |
| `--score_device` | 可选，指定分块算分设备；不写时用第一个 `--devices` 或 `--device`。 |
| `--false_negative_margin` | 如果候选比正例分数还高太多，可能是假负例，默认跳过。 |
| `--max_negative_score_ratio_to_positive` | 可选，限制负例 teacher 分数不能超过最佳正例分数的一定比例，例如 `0.95` 表示只保留 `negative_score <= positive_score * 0.95` 的候选。 |
| `--filter_duplicate_text` | 过滤和正例文本重复、或和本行已选负例文本重复的候选。普通图文任务常用；VisDoc 不建议盲目开。 |
| `--filter_duplicate_media` | 过滤和正例媒体重复、或和本行已选负例媒体重复的候选。会检查规范化路径、文件名 basename，以及本地文件 md5。 |
| `--require_existing_media` | 要求媒体文件真实存在。数据是远程 URI 或延迟生成时可用 `--no-require_existing_media`。 |
| `--write_scores` | 写出 `pos_scores/neg_scores`，方便后续 KD。 |
| `--report_path` | 输出 JSON 统计报告。 |
| `--audit_sample_path` | 输出抽样审计 JSONL，查看每条样本挖到了什么负例。 |

## 结果检查

挖完后至少检查三个文件：

```bash
python -m json.tool $OUTPUT_DIR/report.json | head -n 80
head -n 3 $OUTPUT_DIR/audit.jsonl
head -n 2 $OUTPUT_DIR/train.teacher_hardneg.jsonl
```
