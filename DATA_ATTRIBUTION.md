# Input data and transformations

Source: Google BoolQ validation, immutable Hugging Face revision `35b264d03638db9f4ce671b711558bf7ff0f80d5`.
Dataset card: https://huggingface.co/datasets/google/boolq/blob/35b264d03638db9f4ce671b711558bf7ff0f80d5/README.md
Upstream project: https://github.com/google-research-datasets/boolean-questions
Citation: Clark, Lee, Chang, Kwiatkowski, Collins, and Toutanova, “BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions,” NAACL 2019, https://arxiv.org/abs/1905.10044 .
The pinned dataset card lists Creative Commons Attribution-ShareAlike 3.0: https://creativecommons.org/licenses/by-sa/3.0/ . Retain this attribution and applicable source-license conditions with permitted redistributed derivatives.

Validation row indices are ordered by their stable-ID SHA-256 rank, independently of answer labels. `native/public_promptbank64.jsonl` contains ranks 0–63; `calibration/promptbank64.jsonl` contains ranks 64–127. Question/passage text is retained and a common instruction template is added. The banks are disjoint and include input text/identifiers, not answer/gold fields. File SHA-256 values are in MANIFEST.json and checked by the capture scripts.
The archives contain ordinary-text attention readout and cache-fitting diagnostics.
