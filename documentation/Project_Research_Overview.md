# Patch Attack Verification Research Overview

Last updated: 2026-05-07

## 1. Project Summary

This project studies formal robustness verification for neural networks under localized patch attacks.
The current codebase focuses on MNIST images and an ONNX network stored under ERAN:

- Dataset: `mnist`
- Network: `util/ERAN/tf_verify/models/mnist_convSmallRELU__PGDK.onnx`
- Image shape: `28 x 28`
- Main patch setting used in recent experiments: image index `2`, true label `1`, patch at `(0, 0)`, patch size `10`, patch upper bound `0.75`

The verification question is:

> For a fixed image and a fixed square patch region, can any allowed patch pixel values change the network prediction from the true label to another label?

If no such patch exists for every adversarial label, the image is formally verified as robust for that patch threat model. If a valid counterexample exists, the image is not robust. If the solver times out before proving either result, the label remains unresolved.

The project is therefore a mix of:

- Neural network verification research
- Adversarial robustness certification
- Mixed-integer linear programming formulation research
- Solver performance engineering for Gurobi branch-and-bound
- Empirical experiment design over runtime, timeout schedules, and split strategies

## 2. Threat Model

The attacker controls a square patch inside the image.

For the current MNIST experiments:

- Non-patch pixels are fixed to the original image value.
- Patch pixels can vary independently inside an interval, usually `[0, upper_bound]`.
- The patch is normally placed at the top-left corner `(0, 0)`.
- Patch sizes tested historically include values around `3` through `12`; the current hard case uses patch size `10`.
- Upper bounds tested include values such as `0.70`, `0.75`, `0.80`, `0.85`, `0.90`, `0.95`, and `1.0`.

The generated ERAN input box represents this threat model:

- Fixed pixel: `[pixel_value, pixel_value]`
- Patch pixel: `[0, upper_bound]`

The input box is created by `patch_input_box.py`. ERAN then normalizes the box internally using the dataset mean and standard deviation.

## 3. Verification Backend

The project uses a local copy of ERAN under:

```text
util/ERAN/tf_verify
```

The main verifier wrapper is:

```text
run_verifier.py
```

The wrapper calls ERAN with settings like:

- `--dataset mnist`
- `--domain refinepoly`
- `--complete True`
- `--use_milp True`
- `--timeout_final_milp <seconds>`
- `--label <true_label>`
- `--adv_label <target_label or -1>`
- `--bounds <per-pixel interval list>`
- `--add_bool_constraints True`

ERAN first performs abstract-domain reasoning, then uses MILP refinement for hard labels. Gurobi is the MILP solver. The important solver outcomes in this project are:

| Status | Meaning in this project |
| --- | --- |
| `CUTOFF` / status `6` | The solver proved the label safe through the cutoff objective. |
| `TIME_LIMIT` / status `9` | The label was not resolved before the configured MILP timeout. |
| `INTERRUPTED` / status `11` | Often caused by the custom callback stopping once the objective bound is good enough; the code may still classify the label as verified if the bound proves safety. |
| Counterexample / adversarial example | ERAN found a possible adversarial patch, then the wrapper checks it with the ONNX model. |

## 4. Main Research Problem

The original problem is that some patch verification instances are too slow.

For the hard MNIST case used in recent experiments:

- Image index: `2`
- True label: `1`
- Patch: `(0, 0)`
- Patch size: `10`
- Upper bound: `0.75`
- Hard adversarial labels: `2`, `3`, and `7`

Several labels verify quickly, but labels `2` and `7` can take thousands of seconds. The research question became:

> Can we reformulate the input patch bounds so Gurobi proves the hard labels faster without changing the verified patch domain?

The current direction is to add interval-split structure to selected patch pixels. These splits do not remove any legal patch values when the intervals are adjacent and exhaustive. Instead, they give the MILP solver additional binary variables that can guide branch-and-bound.

## 5. High-Level Pipeline

The current recursive verification pipeline is implemented in:

```text
run_verifier.py::verify_image_with_recursive_timeout_refinement
```

The flow is:

1. Load the MNIST test image and label.
2. Create an input box for the patch threat model.
3. Run one all-label ERAN pass with `adv_label=-1`.
4. Parse the ERAN log into structured per-label outcomes.
5. Keep labels that are already verified or adversarial.
6. Identify labels that timed out.
7. For each timed-out label, extract ERAN's best relaxed fractional example from the log.
8. Convert the ONNX network to a PyTorch model.
9. Compute the gradient of the target-vs-correct class margin with respect to input pixels.
10. Rank patch pixels by gradient influence.
11. Split the top `k` patch pixels, usually `top_k=30`.
12. Rerun ERAN for that single adversarial label only.
13. Repeat until the label resolves or `max_depth` is reached.
14. Write one CSV row per label attempt and save every ERAN log.

This gives a controlled way to study whether interval splits help or hurt each adversarial label.

## 6. Relaxed Example and Gradient-Based Splitting

When Gurobi times out, ERAN logs a relaxed fractional solution:

```text
Best RELAXED (fractional) values: [...]
```

This is not necessarily a valid adversarial patch. It is a solution of a relaxed problem. The project uses it as evidence about where the solver is struggling.

The wrapper computes the margin:

```python
margin = outputs[0, target_class_idx] - outputs[0, correct_class_idx]
```

Then it backpropagates through the converted PyTorch model and ranks patch pixels by gradient magnitude. The selected pixels are the ones whose values most affect the target-vs-correct margin near the relaxed solution.

The important functions are:

- `get_pytorch_model`
- `get_pixel_influence`
- `get_top_k_pixels_in_patch`
- `choose_split_pixels`
- `subdivide_bounds_at_indices`

The split operation repeatedly bisects each selected pixel interval. For example:

```text
Depth 0: [0, 0.75]
Depth 1: [0, 0.375], [0.375, 0.75]
Depth 2: [0, 0.1875], [0.1875, 0.375], [0.375, 0.5625], [0.5625, 0.75]
```

The union of the intervals is still the original `[0, 0.75]` domain. The goal is not to shrink the attack space. The goal is to expose useful structure to the MILP solver.

## 7. MILP Interval-Selection Formulation

The project added a custom input-bound selector formulation inside:

```text
util/ERAN/tf_verify/ai_milp.py::create_model
```

For a pixel `x`, the verifier may have a list of allowed intervals:

```text
[[lb_0, ub_0], [lb_1, ub_1], ..., [lb_n, ub_n]]
```

The MILP must enforce that `x` belongs to one selected interval.

### 7.1 Initial One-Hot Encoding

The first version used one binary variable per interval:

```text
sum(Bool_i) = 1
x <= Bool_0 * ub_0 + Bool_1 * ub_1 + ...
x >= Bool_0 * lb_0 + Bool_1 * lb_1 + ...
```

This is simple but creates many binary variables when a pixel has many intervals.

### 7.2 Logarithmic Encoding

The current formulation uses logarithmic address bits:

- `ceil(log2(N))` binary variables named like `Bit_pixel_i[b]`
- `N` continuous selector variables named like `Aux_pixel_i[j]`
- `sum(aux) == 1`
- Bit-linking constraints connect address bits to active selectors.
- Convex-combination constraints apply the selected interval bounds to `x`.

This trades many binary variables for fewer binary variables plus cheap continuous variables.

For example, with `8` intervals:

- One-hot encoding uses `8` binary variables.
- Logarithmic encoding uses `3` binary variables plus `8` continuous selector variables.

The detailed math explanation lives in:

```text
explanations/log_bool_constraints_for_bounds.md
```

## 8. What We Learned From Split Experiments

The key empirical finding is that splits are not monotonically better.

From the April 24, 2026 investigation for image `2`, patch `(0, 0)`, patch size `10`, upper bound `0.75`:

| Run | Timeout schedule | Total runtime | ERAN attempts | Max depth reached |
| --- | ---: | ---: | ---: | ---: |
| No forced split | `[720000]` | `6771.767s` | `1` | `0` |
| One forced short timeout | `[10, 720000]` | `5175.025s` | `4` | `1` |
| Two forced short timeouts | `[10, 10, 720000]` | `8094.688s` | `7` | `2` |
| Three forced short timeouts | `[10, 10, 10, 720000]` | `6151.716s` | `10` | `3` |

Per-label behavior was different:

| Label | Best observed behavior |
| ---: | --- |
| `2` | Depth `3` was fastest in that investigation. |
| `3` | Usually solved near the root; splitting did not matter much. |
| `7` | Depth `1` was fastest; deeper splits were slower. |

The conclusion is:

- Splitting can reduce Gurobi's branch-and-bound tree.
- Splitting also increases model size and per-node LP cost.
- Runtime depends on the balance between fewer explored nodes and more expensive nodes.
- A globally fixed split depth is probably not optimal.

The investigation journal is:

```text
journal/20260425_split_runtime_investigation.md
```

## 9. Important Upgrades Already Added

This section summarizes the different upgrades made during the project.

### 9.1 Structured ERAN Log Parsing

`run_verifier.py` now parses ERAN logs into structured details:

- Failed labels
- Per-label MILP status
- Final return value
- Possible adversarial examples
- Relaxed fractional examples
- Per-label runtimes
- Candidate and adversarial label mappings

This made it possible to build reliable CSV outputs instead of manually reading logs.

### 9.2 Per-Pixel Bounds

The code now supports a full per-pixel bounds list aligned with ERAN's flattened input order.

Instead of giving every patch pixel the same interval list, the recursive flow can split only selected pixels. This is necessary for gradient-guided refinement.

Relevant files:

- `patch_input_box.py`
- `utils.py`
- `run_verifier.py`
- `util/ERAN/tf_verify/__main__.py`
- `util/ERAN/tf_verify/ai_milp.py`

### 9.3 Recursive Timeout Refinement

The current main algorithm runs timed-out labels again after splitting selected patch pixels. This upgrade changed the project from one-shot verification into an iterative refinement system.

Important behavior:

- Initial run checks all labels.
- Reruns target one adversarial label at a time.
- Each timed-out label has its own refinement state.
- The pipeline stops a label when it is verified, adversarial, or reaches max depth.
- If any label is truly adversarial, the image is not robust and the run can stop early.

### 9.4 Gradient-Based Split Selection

The project added a way to choose split pixels from the relaxed MILP solution rather than splitting arbitrary or random pixels.

The idea is:

- Use the relaxed example from the timeout.
- Compute the target-vs-correct margin gradient.
- Rank patch pixels by influence.
- Split the top `k` pixels.

This gives a label-sensitive split strategy.

### 9.5 Logarithmic Boolean Encoding

The old one-hot interval encoding was replaced by a logarithmic address-bit formulation. This reduces the number of binary variables needed when one pixel has many intervals.

This is one of the central formulation upgrades in the project.

### 9.6 Singleton Selector Skipping

A recent deterministic optimization skips selector constraints for pixels that have only one interval.

Without this upgrade, even an unsplit patch pixel with a single interval could get redundant selector variables and constraints.

The upgraded behavior:

- If `skip_singleton_bounds=True` and a pixel has one interval, no selector variables are created.
- The input variable bounds are tightened to the singleton interval when needed.
- This preserves the semantics while reducing model size.

Current default in the project runner:

```text
SKIP_SINGLETON_BOUNDS = True
```

### 9.7 Split-Bit Branch Priority

The project added an explicit Gurobi branch-priority option for split-bit variables.

The research motivation is that split variables are chosen from gradient information and may represent useful branching decisions. If Gurobi branches on unrelated network binaries first, it may not benefit from the split structure soon enough.

Current controls:

```text
enable_split_bit_branch_priority
split_bit_branch_priority
```

When enabled, every `Bit_pixel_*` variable gets:

```python
bin_vars[b].BranchPriority = split_bit_branch_priority
```

The current runner uses:

```text
ENABLE_SPLIT_BIT_BRANCH_PRIORITY = True
SPLIT_BIT_BRANCH_PRIORITY = 1000
```

### 9.8 Detailed CSV Schema

The CSV output now records one row per label attempt.

Key fields include:

- `image_index`
- `true_label`
- `adv_label`
- `patch_x`, `patch_y`, `patch_size`
- `upper_bound`
- `timeout_milp_s`
- `skip_singleton_bounds`
- `enable_split_bit_branch_priority`
- `split_bit_branch_priority`
- `split_indices`
- `run_scope`
- `global_attempt_index`
- `label_attempt_index`
- `current_depth`
- `status_code`
- `status_name`
- `final_outcome`
- `iteration_runtime_seconds`
- `attempt_runtime_seconds`
- `parent_total_run_time_seconds`
- `log_path`

The schema is defined in:

```text
utils.py::RECURSIVE_TIMEOUT_REFINEMENT_COLUMNS
```

### 9.9 Per-Run Log Directories

The project now writes logs into timestamped daily run folders:

```text
logs/<date>/<run_id>/<attempt>_<scope>_eran_run.log
```

Examples:

```text
logs/20260506/20260506_181950_run_01_ub_0p75_timeouts_720000/000_all_labels_eran_run.log
logs/20260506/20260506_181950_run_02_ub_0p75_timeouts_10_720000/001_adv_label_2_eran_run.log
```

This makes it much easier to connect CSV rows back to the exact ERAN/Gurobi logs.

### 9.10 Controlled Experiment Runners

The main batch runner is:

```text
scripts/run_multiple_till_sol.py
```

It currently runs the image `2`, patch `(0, 0)`, size `10`, `upper_bound=0.75`, and several timeout schedules.

The controlled formulation-comparison runner is:

```text
scripts/run_split_priority_experiments.py
```

It compares:

- Current formulation
- Singleton skipping only
- Singleton skipping plus branch priority
- Singleton skipping plus branch priority plus a fixed core-24 split set

The core-24 fixed split set came from high overlap among the top-gradient pixels for labels `2`, `3`, and `7`.

### 9.11 Plotting and Runtime Analysis

The project includes plotting and flattening helpers for comparing per-label runtimes:

- `scripts/plot_per_label_runtime_comparison.py`
- `scripts/flatten_recursive_timeout_refinement_per_label.py`

The plots under `images/` are used to compare benchmark runs against recursive split runs across labels and upper bounds.

## 10. Current Experiment State As Of This Document

The latest visible batch artifacts are based on run id:

```text
20260506_181950
```

The latest CSV at inspection time was:

```text
csv/recursive_timeout_refinement_20260506_181950.csv
```

At the time this document was written:

- The CSV had completed rows for the long no-forced-split run.
- That run used `upper_bound=0.75`, `skip_singleton_bounds=True`, `enable_split_bit_branch_priority=True`, and `split_bit_branch_priority=1000`.
- It verified all labels with total runtime about `6535.205s`.
- The next schedule `[10, 720000]` had started.
- The initial all-label attempt timed out on labels `2`, `3`, and `7`.
- A long single-label rerun for adversarial label `2` was active.
- The active rerun log showed `selector_pixels=30`, `split_bit_vars=60`, `aux_selector_vars=60`, `singleton_intervals_skipped=70`, and `branch_priority_vars=60`.

This means the new branch-priority formulation is wired into the active experiment, but the full runtime conclusion for the branch-priority upgrade still depends on completing and comparing the long runs.

## 11. How To Read Results

For any experiment, start with the CSV row group.

Use these columns first:

- `parent_status`: overall image result for that run
- `adv_label`: target adversarial label
- `current_depth`: how many split refinements have been applied for that label
- `status_name`: Gurobi status name
- `final_outcome`: normalized project outcome
- `iteration_runtime_seconds`: per-label runtime, when available
- `attempt_runtime_seconds`: full ERAN attempt runtime
- `split_indices`: flattened pixel indices that were split before that attempt
- `log_path`: exact ERAN/Gurobi log to inspect

Then inspect the matching log file for:

- `Input split selector settings`
- `Input split selector stats`
- Gurobi original and presolved model size
- Number of binary variables
- Root relaxation objective
- Explored nodes
- Simplex iterations
- Final bound and final status
- `Best RELAXED (fractional) values`

The most useful comparisons are per adversarial label and per depth. Total runtime alone can hide the fact that one label improved while another got worse.

## 12. Research Conclusions So Far

The strongest conclusions so far are:

1. Patch robustness verification can be dominated by a small number of hard adversarial labels.
2. Relaxed timeout solutions provide useful information for choosing split pixels.
3. Splitting important pixels can help Gurobi by changing the branch-and-bound search tree.
4. More splitting is not automatically better.
5. Logarithmic interval selection is a better fit than one-hot interval selection as interval counts grow.
6. Singleton selector constraints are redundant and should be skipped.
7. Branch priority is a plausible way to make the solver exploit the split variables earlier, but it needs complete experimental evidence.
8. Runtime should be evaluated per label, not only per full image run.

## 13. Open Questions And Next Steps

The main open questions are:

- Does split-bit branch priority consistently reduce explored nodes or total runtime for the hard labels?
- Is the core-24 fixed split set enough to preserve most of the benefit with less model overhead?
- Should the system choose different split depths per label?
- Should the system run short probes at multiple depths, then choose the most promising formulation for the long solve?
- Should future experiments compare against fresh single-label no-split baselines, to separate the effect of splitting from the effect of all-label model reuse?
- Can selected split pixel order be passed into Gurobi as graded priorities instead of one uniform priority for every split bit?

Recommended near-term experiment:

1. Finish the `20260506_181950` batch run.
2. Compare the `[720000]` no-forced-split run against `[10, 720000]`, `[10, 10, 720000]`, and `[10, 10, 10, 720000]`.
3. Compare each hard label separately: `2`, `3`, and `7`.
4. Extract solver nodes, simplex iterations, presolved binary counts, and runtime from each log.
5. Run `scripts/run_split_priority_experiments.py` when a full controlled formulation comparison is desired.

## 14. File Map

| Path | Purpose |
| --- | --- |
| `run_verifier.py` | Main wrapper around ERAN, log parsing, recursive timeout refinement, gradient split selection. |
| `patch_input_box.py` | Builds patch input boxes and per-pixel bounds structures. |
| `utils.py` | CSV schema, log directory helpers, bounds subdivision helpers, CSV append logic. |
| `config.py` | Project paths, ERAN paths, MNIST data path, network path, Gurobi license path. |
| `util/ERAN/tf_verify/ai_milp.py` | Modified ERAN MILP model creation, input interval selector constraints, singleton skipping, branch priority. |
| `util/ERAN/tf_verify/__main__.py` | Modified ERAN CLI args, bounds normalization, analyzer invocation. |
| `util/ERAN/tf_verify/config.py` | ERAN-side default config values for custom options. |
| `scripts/run_multiple_till_sol.py` | Main batch runner for upper bounds and timeout schedules. |
| `scripts/run_split_priority_experiments.py` | Controlled runner for singleton skipping and branch priority variants. |
| `scripts/plot_per_label_runtime_comparison.py` | Plotting helper for benchmark vs split runtimes. |
| `explanations/log_bool_constraints_for_bounds.md` | Mathematical explanation of logarithmic interval selection. |
| `journal/20260425_split_runtime_investigation.md` | Detailed investigation notes and empirical findings about split runtime behavior. |
| `csv/` | Structured experiment outputs. |
| `logs/` | Raw ERAN/Gurobi logs by date and run id. |
| `images/` | Runtime comparison plots and patch robustness figures. |

## 15. Short Version For Presentations

This project verifies whether MNIST classifiers are robust to localized patch attacks using ERAN plus Gurobi MILP. The hard cases are not usually all labels; runtime is dominated by a few adversarial labels. We added a recursive timeout-refinement loop that uses the relaxed MILP solution from a timeout to identify high-gradient patch pixels, splits those pixels into intervals, and reruns verification for the timed-out label. The split intervals preserve the same patch attack domain, but introduce structured binary variables that can help branch-and-bound.

The main formulation upgrade is a logarithmic interval selector encoding, using `ceil(log2(N))` binary variables instead of one binary per interval. Recent upgrades also skip redundant singleton selectors, log detailed per-label metrics, and optionally assign high Gurobi branch priority to split-bit variables. Early evidence shows that splits can substantially speed up some hard labels, but the effect is label- and depth-dependent. The current research is about making that speedup more predictable.
