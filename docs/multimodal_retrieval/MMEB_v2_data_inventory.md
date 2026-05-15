# MMEB v2 Data Inventory

Last updated: 2026-03-24

## Goal

This document records the MMEB v2 evaluation coverage, the public training sources already identified in local references, and the target Nexus data formats that the conversion tooling should emit.

The local source of truth used here is `VLM2Vec`, especially:

- `experiments/report_score_v2.py`
- `experiments/public/eval/*.yaml`
- `experiments/public/train/*.yaml`
- `src/constant/dataset_hf_path.py`
- `src/constant/dataset_hflocal_path.py`

There is also a generated machine-readable export at:

- `docs/multimodal_retrieval/MMEB_v2_manifest.json`

## Scoreboard Granularity

MMEB v2 is aggregated into 3 modalities and 78 tasks:

- `image`
- `video`
- `visdoc`

The most reliable local scoring reference is `VLM2Vec/experiments/report_score_v2.py`.

## Evaluation Coverage

### Image

Task families:

- `Image-CLS`
- `Image-QA`
- `i2t`
- `t2i`
- `i2i / composed retrieval`
- `VG`

Datasets:

- `ImageNet-1K`
- `N24News`
- `HatefulMemes`
- `VOC2007`
- `SUN397`
- `Place365`
- `ImageNet-A`
- `ImageNet-R`
- `ObjectNet`
- `Country211`
- `OK-VQA`
- `A-OKVQA`
- `DocVQA`
- `InfographicsVQA`
- `ChartQA`
- `Visual7W`
- `ScienceQA`
- `VizWiz`
- `GQA`
- `TextVQA`
- `MSCOCO_i2t`
- `VisualNews_i2t`
- `VisDial`
- `MSCOCO_t2i`
- `VisualNews_t2i`
- `WebQA`
- `EDIS`
- `Wiki-SS-NQ`
- `CIRR`
- `NIGHTS`
- `OVEN`
- `FashionIQ`
- `MSCOCO`
- `RefCOCO`
- `RefCOCO-Matching`
- `Visual7W-Pointing`

### Video

Task families:

- `Video-CLS`
- `Video-RET`
- `Video-MRET`
- `Video-QA`

Datasets:

- `SmthSmthV2`
- `HMDB51`
- `UCF101`
- `Kinetics-700`
- `Breakfast`
- `MSR-VTT`
- `MSVD`
- `DiDeMo`
- `YouCook2`
- `VATEX`
- `QVHighlight`
- `Charades-STA`
- `MomentSeeker`
- `Video-MME`
- `NExTQA`
- `EgoSchema`
- `MVBench`
- `ActivityNetQA`

### Visdoc

Task families:

- `ViDoRe classic`
- `ViDoRe v2`
- `VisRAG`
- `ViDoSeek / MMLongBench`

Datasets:

- `ViDoRe_arxivqa`
- `ViDoRe_docvqa`
- `ViDoRe_infovqa`
- `ViDoRe_tabfquad`
- `ViDoRe_tatdqa`
- `ViDoRe_shiftproject`
- `ViDoRe_syntheticDocQA_artificial_intelligence`
- `ViDoRe_syntheticDocQA_energy`
- `ViDoRe_syntheticDocQA_government_reports`
- `ViDoRe_syntheticDocQA_healthcare_industry`
- `ViDoRe_esg_reports_human_labeled_v2`
- `ViDoRe_biomedical_lectures_v2`
- `ViDoRe_biomedical_lectures_v2_multilingual`
- `ViDoRe_economics_reports_v2`
- `ViDoRe_economics_reports_v2_multilingual`
- `ViDoRe_esg_reports_v2`
- `ViDoRe_esg_reports_v2_multilingual`
- `VisRAG_ArxivQA`
- `VisRAG_ChartQA`
- `VisRAG_MP-DocVQA`
- `VisRAG_SlideVQA`
- `VisRAG_InfoVQA`
- `VisRAG_PlotQA`
- `ViDoSeek-page`
- `ViDoSeek-doc`
- `MMLongBench-page`
- `MMLongBench-doc`

## Public Training Sources

### Image training sources

Primary public source:

- `TIGER-Lab/MMEB-train`

Public subsets already used in the VLM2Vec public recipe:

- `ImageNet_1K`
- `N24News`
- `HatefulMemes`
- `VOC2007`
- `SUN397`
- `OK-VQA`
- `A-OKVQA`
- `DocVQA`
- `InfographicsVQA`
- `ChartQA`
- `Visual7W`
- `VisDial`
- `CIRR`
- `VisualNews_t2i`
- `VisualNews_i2t`
- `MSCOCO_t2i`
- `MSCOCO_i2t`
- `NIGHTS`
- `WebQA`
- `MSCOCO`

Notes:

- Public config uses `original` split.
- The local VLM2Vec README notes that `diverse_instruction` is also available and is the stronger recommendation for later experiments.

### Video training sources

Primary public source:

- `ShareGPTVideo/train_video_and_instruction`

Public recipe entries:

- `video_caption_300k`
- `video_caption_300k-video`
- `video_qa_240k`

Expected raw format in VLM2Vec references:

- JSON or JSONL
- fields like `id`, `video`, `conversations`
- frame extraction stored under a configurable `video_frame_basedir`

### Visdoc training sources

Primary public sources:

- `vidore/colpali_train_set`
- `openbmb/VisRAG-Ret-Train-In-domain-data`

Notes:

- These are used as generic visual-document retrieval training sources rather than one train source per visdoc benchmark subset.

## Target Nexus Formats

### Training format

Each Nexus training row should follow the multimodal retrieval format:

```json
{
  "query": {"text": "...", "image_path": "..."},
  "pos": [{"text": "...", "image_path": "..."}],
  "neg": [{"text": "...", "image_path": "..."}],
  "metadata": {"dataset_name": "..."},
  "base_dir": "/optional/base/dir",
  "image_root": "/optional/image/root",
  "video_root": "/optional/video/root"
}
```

Supported item fields in the current Nexus multimodal path:

- `text`
- `title`
- `image`
- `images`
- `image_path`
- `image_paths`
- `pages`
- `video`
- `video_path`
- `videos`
- `video_paths`
- `video_frames`

### Evaluation format

Each Nexus local evaluation dataset should contain:

- `corpus.jsonl`
- `test_queries.jsonl`
- `test_qrels.jsonl`

Media roots can now be supplied separately at evaluation time:

- `media_root`
- `image_root`
- `video_root`

## Conversion Priorities

1. `MMEB-train` image subsets to Nexus train JSONL.
2. `ShareGPTVideo/train_video_and_instruction` to Nexus train JSONL with raw-video or frame-based options.
3. `ViDoRe` and `VisRAG` train sources to Nexus train JSONL.
4. VLM2Vec pair-style evaluation exports to Nexus local retrieval datasets.

## Notes For Data Collection

- Use this document together with the conversion scripts under `tools/multimodal_retrieval/`.
- Keep raw media roots configurable instead of rewriting everything to absolute paths.
- Prefer isolated environments for download and preprocessing jobs.
