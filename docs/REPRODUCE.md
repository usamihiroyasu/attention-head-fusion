# Reproduce the archived diagnostics

Run commands from the repository root and write new outputs under `runs/`.

## Portable proof (Python standard library)

```sh
mkdir -p runs/proof
python3 certificate/portable_interval_minor17.py --input-npz certificate/certificate_input17.npz --output runs/proof/replay.json
python3 certificate/verify_stored_rank.py --enclosure runs/proof/replay.json --preconditioner certificate/ROOT_EXACT_RATIONAL_RANK_CHECK.json
```
Expected states: `CERTIFIED_NONZERO_MINOR`, then `PASS_EXACT_RATIONAL_RANK17`. This replay from the supplied FP32 constants has been executed successfully. It certifies the real-exponential map on 17 fixed queries, not full floating-point model execution or numerical SVD error-floor magnitudes.

## Controlled CPU diagnostic

Use NumPy 2.4.4 as recorded in `environment/controlled.txt`. Copy before running because the original script writes RESULTS.json beside itself.

```sh
mkdir -p runs
cp -R controlled runs/controlled
python3 runs/controlled/controlled.py
```
The reference measurements are `controlled/RESULTS.json` and `controlled/SUMMARY.csv`. There are 36 method/bank conditions; the 27 exact-method conditions have maximum absolute error at most 7.0344e-13. Timing is descriptive and is not expected to reproduce bit-for-bit across hardware.

## Native capture and learned caches (CUDA)

The archived environment used Python 3.12.3 with the versions in `environment/native.txt`. Obtain the six files listed in `assets/ASSETS.json` at their exact pinned revisions and place them in `assets/checkpoint/`; check their hashes. No model download is performed by this package. The workers require CUDA and assert the pinned Transformers/PEFT versions. CUDA is required for this pipeline.

```sh
mkdir -p runs/native runs/learned runs/calibration
python3 native/worker.py --assets assets/checkpoint --prompts native/public_promptbank64.jsonl --output runs/native/results.json
python3 learned/capture_worker.py --assets assets/checkpoint --prompts native/public_promptbank64.jsonl --output runs/learned/capture_receipt.json
python3 learned/learn_worker.py --capture runs/learned --output runs/learned/LEARN_RESULTS.json --fitter learned/fitter.py --fitter-sha fe69094fa59624f819ba53d8e6d065901defcc37ceb8b65d529c58d518a64639
python3 calibration/capture_candidate.py --assets assets/checkpoint --prompts calibration/promptbank64.jsonl --prompts-sha256 3e829e50d53033e8ffd71d8f739e635f81d0c103796293293d6e00b14f56ad06 --exclude-prompts native/public_promptbank64.jsonl --output runs/calibration/capture_receipt.json
python3 calibration/learn_candidate.py --capture runs/calibration --output runs/calibration/LEARN_RESULTS.json --fitter calibration/fitter.py --fitter-sha fe69094fa59624f819ba53d8e6d065901defcc37ceb8b65d529c58d518a64639
```
The capture stage creates its relative `capture/cal` and `capture/held` arrays under the output parent. The learner reads that parent's `capture_receipt.json` and writes FP32 records under `records/`. Do not rerun into an existing output directory. Each original learner uses 512 updates and four fixed starts, fixes selectors using calibration exports, and only then opens held-out arrays. The follow-up fixes all eight selectors before the common final-16-query evaluation.

## Compare reference results

Inspect `native/AGGREGATE.json`, `learned/ROOT_AGGREGATE.json`, and `calibration/ROOT_AGGREGATE.json` together with their full result files. Compare equal-context macro NRMSE, allocated bytes, selectors, and all-start diagnostics; reference outcomes include the original mixed rankings. Numerical results can vary with device and library versions. Compare errors, allocations, and selector counts with the archived reference outputs. The portable rank certificate, prompt-bank digests, and allocation formulas can be checked independently of model execution.
