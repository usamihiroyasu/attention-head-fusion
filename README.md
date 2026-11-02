# Exact Fusion of Softmax Attention Heads Can Require Exponentially Many Atoms

Hiroyasu Usami · Keisuke Hara · Ayato Tsuboi · Naohiko Matsuda

This repository accompanies the manuscript and provides its implementations, archived results, protocols, and certificate inputs.

The paper asks how many explicit atoms are needed when independently normalized attention heads are replaced by one softmax. For the constructed family, `d` two-atom heads use `2d` head–atom terms, while the minimum equivalent single-head representation needs `2^d` atoms: **24 terms versus 4096 atoms when `d = 12`**.

## Repository layout

| Directory | Contents |
| --- | --- |
| `certificate/` | Interval replay and exact rational verification of the stored rank-17 certificate |
| `controlled/` | Fixed-cache FP64 diagnostics and all 36 method/bank conditions |
| `native/` | Native-cache scalar fusion diagnostic and reference results |
| `learned/` | Equal-allocated-byte vector-cache fitting, all starts, and selected outcomes |
| `calibration/` | Calibration-policy comparison on a second, disjoint 64-context bank |
| `protocol_history/` | Frozen specifications and dated protocol amendments |

## Quick start: verify the certificate

Only the Python standard library is required. Run from the repository root:

```sh
mkdir -p runs/proof
python3 certificate/portable_interval_minor17.py --input-npz certificate/certificate_input17.npz --output runs/proof/replay.json
python3 certificate/verify_stored_rank.py --enclosure runs/proof/replay.json --preconditioner certificate/ROOT_EXACT_RATIONAL_RANK_CHECK.json
```

Expected results are `CERTIFIED_NONZERO_MINOR` and `PASS_EXACT_RATIONAL_RANK17`. This check concerns the real-exponential map on the 17 supplied queries.

## Reproducing the experiments

[Reproduction instructions](docs/REPRODUCE.md) specify the controlled CPU diagnostic and the CUDA capture/fitting pipeline. Pinned package versions are recorded in [`environment/`](environment/), and exact external asset identifiers and hashes are listed in [`assets/ASSETS.json`](assets/ASSETS.json). Write new outputs under `runs/` to preserve the archived reference results.

The scalar identity diagnostic, finite-query vector certificate, and learned approximation experiments address different representation questions. The archived results include every reported condition and budget-dependent ranking reversals.

## Data and reuse

Input banks retain the attribution and CC BY-SA 3.0 conditions described in [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). External model weights are obtained separately. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE_STATUS.md](LICENSE_STATUS.md) for the applicable terms. No reuse license has yet been assigned to author-owned code.

File hashes are recorded in [MANIFEST.json](MANIFEST.json). Citation metadata are available in [CITATION.cff](CITATION.cff).
