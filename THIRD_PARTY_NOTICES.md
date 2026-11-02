# External assets and dependencies

- BoolQ input data: Clark et al. (2019), CC BY-SA 3.0 as recorded by the pinned dataset card; see DATA_ATTRIBUTION.md.
- CODI GPT-2 checkpoint: `zen-E/CODI-gpt2`, fixed revision in assets/ASSETS.json; its model card records MIT. Weights are fetched separately and are not relicensed by this package.
- GPT-2 configuration/tokenizer: `openai-community/gpt2`, fixed revision in assets/ASSETS.json; retain the upstream repository terms when downloading those assets.
- Scientific dependencies are Python, NumPy, PyTorch, Transformers, and PEFT. Their licenses remain those of their upstream distributions; no dependency package is bundled here.
- The portable certificate scripts use only the Python standard library. Code license selection for this candidate remains pending as specified in LICENSE_STATUS.md.
