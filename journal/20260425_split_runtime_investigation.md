# Split Runtime Investigation Journal

## Rules for Continuing This Journal

- Keep this file in append-style order: newest entries go at the bottom unless updating the current conclusion section.
- Every entry must start with a UTC timestamp and model name in this format: `YYYY-MM-DD HH:MM:SS UTC | Model: <model>`.
- Record what was inspected, why it matters, and any concrete conclusion or open question.
- Prefer exact file paths, run ids, labels, timeouts, node counts, and runtimes over vague summaries.
- Separate observations from hypotheses. Mark hypotheses as provisional until backed by log evidence.
- Do not delete prior entries; if a conclusion changes, add a correction entry with the new evidence.

## Investigation Target

We are comparing these four runs for image index 2, patch `(0, 0)`, patch size 10, upper bound 0.75, max depth 4:

- `logs/20260424/20260424_134749_run_01_ub_0p75_timeouts_720000`
- `logs/20260424/20260424_134749_run_02_ub_0p75_timeouts_10_720000`
- `logs/20260424/20260424_134749_run_03_ub_0p75_timeouts_10_10_720000`
- `logs/20260424/20260424_134749_run_04_ub_0p75_timeouts_10_10_10_720000`

The question is why doing more recursive image splits can make verification against some adversarial labels faster, even though splitting creates more subranges to check.

## Entries

### 2026-04-25 13:42:57 UTC | Model: Codex GPT-5

Started by reading `scripts/run_multiple_till_sol.py` and listing the requested log folders. The runner schedules the same image and upper bound four times, changing only the MILP timeout schedule:

- Run 01: `[720000]`
- Run 02: `[10, 720000]`
- Run 03: `[10, 10, 720000]`
- Run 04: `[10, 10, 10, 720000]`

Initial observation: the later runs intentionally force shallow attempts to time out quickly, then retry unresolved labels after recursive splitting. The core comparison should therefore be per adversarial label, not just total folder runtime, because labels can resolve at different depths.

Next checks planned:

- Find the CSV or summary output for these runs.
- Extract per-label runtime, depth, status, and split metadata from each log.
- Compare Gurobi model and presolve stats for the same adversarial label across depths.
- Compare Gurobi node progress over time to see whether smaller split boxes produce easier branch-and-bound behavior despite creating more boxes.

### 2026-04-25 13:50:09 UTC | Model: Codex GPT-5

Found the detailed CSV for these runs:

- `csv/recursive_timeout_refinement_20260424_134749.csv`

The hard labels are adversarial labels 2, 3, and 7. Labels 0, 4, 5, 6, 8, and 9 verify quickly during the initial all-label attempt. Label 3 is only mildly hard; labels 2 and 7 dominate runtime.

Run totals from the CSV:

| Run | Timeout schedule | Total runtime | ERAN attempts | Max depth reached |
| --- | --- | ---: | ---: | ---: |
| Run 01 | `[720000]` | 6771.767s | 1 | 0 |
| Run 02 | `[10, 720000]` | 5175.025s | 4 | 1 |
| Run 03 | `[10, 10, 720000]` | 8094.688s | 7 | 2 |
| Run 04 | `[10, 10, 10, 720000]` | 6151.716s | 10 | 3 |

Per-hard-label successful long attempt runtimes:

| Label | Run 01 depth 0 | Run 02 depth 1 | Run 03 depth 2 | Run 04 depth 3 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 4321.419s | 3511.873s | 4607.958s | 3114.820s |
| 3 | 27.525s | 45.360s | 39.354s | 40.503s |
| 7 | 2387.126s | 1543.017s | 3296.719s | 2771.598s |

Initial conclusion: splitting does not monotonically improve runtime. Run 02 is fastest because depth 1 improves both dominant labels, 2 and 7. Run 03 is slow because depth 2 makes both labels 2 and 7 worse. Run 04 beats the no-split run mainly because label 2 becomes much faster at depth 3, but label 7 is slower than no-split.

Short timeout attempts cost more than 10 seconds wall time. Single-label 10s reruns take about 24-26 seconds each because after Gurobi reaches the 10s MILP limit, the code runs a relaxed model to extract `Best RELAXED (fractional) values` for choosing the next split. This overhead matters in runs 03 and 04.

Model/presolve size observations:

| Depth | Interval shape for selected pixels | Num binary vars before presolve | Presolved columns | Presolved binary vars |
| ---: | --- | ---: | ---: | ---: |
| 0 | no selected split pixels | 580 | 1882 | 563 |
| 1 | 30 pixels x 2 intervals | 640 | 1908 | 591 |
| 2 | same 30 pixels x 4 intervals | 670 | 2058 | 621 |
| 3 | same 30 pixels x 8 intervals | 700 | 2208 | 651 |

Important implementation detail: the recursive splitter repeatedly splits the same 30 high-gradient pixels for each label in these runs. It does not spread to 60 or 90 different pixels. The bounds counts parsed from rerun logs are:

- Depth 1: `{1: 754, 2: 30}`
- Depth 2: `{1: 754, 4: 30}`
- Depth 3: `{1: 754, 8: 30}`

This means deeper runs add one more selector bit for each of the same 30 pixels. The mathematical input domain still covers the same `[0, 0.75]` range for those pixels because the intervals are adjacent and exhaustive. The speed change is therefore likely a branch-and-bound/formulation effect, not a smaller external verification domain. Gurobi gets extra input selector binaries that partition the domain and may help prove cutoff faster for some labels, but they also enlarge the formulation and can hurt search.

### 2026-04-25 13:51:23 UTC | Model: Codex GPT-5

Compared Gurobi node progress for the dominant hard labels. Root relaxations are almost unchanged across depths, so the speed differences are not explained by a stronger root LP:

| Label | Depth | Root objective | Final solver runtime | Explored nodes | Simplex iterations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | -1.869134 | 4321.38s | 6146 | 6384170 |
| 2 | 1 | -1.867331 | 3498.24s | 5318 | 5282238 |
| 2 | 2 | -1.869225 | 4594.63s | 6841 | 7518053 |
| 2 | 3 | -1.869137 | 3101.46s | 3967 | 4622023 |
| 7 | 0 | -1.663233 | 2387.09s | 4985 | 3407700 |
| 7 | 1 | -1.661463 | 1529.43s | 3259 | 2026898 |
| 7 | 2 | -1.665423 | 3283.80s | 5258 | 4337197 |
| 7 | 3 | -1.663235 | 2758.43s | 4544 | 3521048 |

Gurobi progress threshold observations:

- Label 2 depth 3 reached `BestBd >= -0.75` at 861s, much earlier than depth 0 at 2280s and depth 1 at 2018s. It finished after fewer explored nodes than all other label-2 variants.
- Label 2 depth 2 reached `BestBd >= -0.75` at 641s but still finished slowest, so early bound improvement alone is not enough; later branch-and-bound search can still be worse.
- Label 7 depth 1 is faster mainly because it explores fewer nodes than depth 0, 2, or 3.
- Label 7 depth 2 and depth 3 do not improve the root objective and require more/slower search than depth 1.

Current hypothesis: splits help only when the extra input selector binaries give Gurobi a better branching/pruning structure for the specific label. They can reduce the node count dramatically, as with label 2 depth 3 and label 7 depth 1. But because they also add variables and change the branch-and-cut tree, they can make the search worse, as with label 2 depth 2 and label 7 depth 2/3.

Important comparison caveat: Run 01 is an all-label run with one Gurobi model reused across labels by changing the objective. The recursive reruns are fresh single-label ERAN/Gurobi processes. Therefore, comparing Run 01 depth 0 directly to Run 02/03/04 final single-label reruns mixes two effects: no-split vs split, and all-label model reuse vs fresh single-label solve. A cleaner experiment would add fresh single-label long-timeout baselines at depth 0 for labels 2, 3, and 7.

Status caveat: label 2 in Run 01 has status `INTERRUPTED` but `final_outcome=verified`. This is expected from the custom callback in `util/ERAN/tf_verify/ai_milp.py`: it terminates when the objective bound exceeds `0.01`. The code then treats positive `objbound` as verified even if Gurobi status is 11.

### 2026-04-25 14:02:10 UTC | Model: Codex GPT-5

Read `explanations/log_bool_constraints_for_bounds.md` and the implementation in `util/ERAN/tf_verify/ai_milp.py:create_model`. The implementation uses a logarithmic address encoding:

- `num_bits = ceil(log2(bounds_len))`
- `Bit_pixel_i[...]` are binary address variables.
- `Aux_pixel_i[...]` are continuous selectors.
- `sum(aux) == 1`, bit-link constraints, and convex-combination upper/lower bound constraints connect the selected interval to input variable `x_i`.

Also checked `run_verifier.py:verify_image_with_recursive_timeout_refinement`. The input box file is recreated as the same full patch box for every attempt. The recursive split state is passed separately as `bounds` into ERAN/Gurobi. Therefore, these depth comparisons are intentionally testing the same outer patch domain with different internal interval-selector formulations.

Model-size deltas by depth:

| Depth | Selected-pixel intervals | Original rows | Original cols | Original binary vars | Presolved cols | Presolved binary vars |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 patch pixels with 1 interval | 128703 | 8682 | 580 | 1878 | 561 |
| 1 | 30 selected pixels with 2 intervals | 128733 | 8772 | 640 | 1908 | 591 |
| 2 | same 30 selected pixels with 4 intervals | 128763 | 8862 | 670 | 2058 | 621 |
| 3 | same 30 selected pixels with 8 intervals | 128793 | 9012 | 700 | 2208 | 651 |

Notes on this table:

- Each depth adds only 30 original rows, because each selected pixel gets one more bit-link equation when the number of intervals doubles.
- Depth 2 and 3 add exactly 30 more binary vars each, matching one additional address bit for each of the 30 selected pixels.
- Depth 1 shows +60 binary vars versus depth 0, not just +30. The code only explicitly adds one address bit per two-interval selected pixel. A likely explanation is that Gurobi recognizes one selector variable per two-interval pixel as binary/integral because it is directly linked to the address bit. This should be verified by dumping variable names/types if exact accounting matters.

More detailed node metrics for the benchmark and final high-timeout attempt at each depth:

| Label | Depth | Solver sec | Nodes | Nodes/sec | Sec/node | Simplex/node | Root objective | Presolved cols/bin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 0 | 4321.38 | 6146 | 1.422 | 0.703 | 1039 | -1.869134 | 1878 / 561 |
| 2 | 1 | 3498.24 | 5318 | 1.520 | 0.658 | 993 | -1.867331 | 1908 / 591 |
| 2 | 2 | 4594.63 | 6841 | 1.489 | 0.672 | 1099 | -1.869225 | 2058 / 621 |
| 2 | 3 | 3101.46 | 3967 | 1.279 | 0.782 | 1165 | -1.869137 | 2208 / 651 |
| 3 | 0 | 27.48 | 1 | 0.036 | 27.480 | 12423 | -0.614697 | 1878 / 561 |
| 3 | 1 | 31.28 | 1 | 0.032 | 31.280 | 12697 | -0.612698 | 1908 / 591 |
| 3 | 2 | 26.59 | 1 | 0.038 | 26.590 | 11297 | -0.614702 | 2058 / 621 |
| 3 | 3 | 27.36 | 1 | 0.037 | 27.360 | 20584 | -0.614702 | 2208 / 651 |
| 7 | 0 | 2387.10 | 4985 | 2.088 | 0.479 | 684 | -1.663233 | 1878 / 561 |
| 7 | 1 | 1529.44 | 3259 | 2.131 | 0.469 | 622 | -1.661463 | 1908 / 591 |
| 7 | 2 | 3283.80 | 5258 | 1.601 | 0.625 | 825 | -1.665423 | 2058 / 621 |
| 7 | 3 | 2758.44 | 4544 | 1.647 | 0.607 | 775 | -1.663235 | 2208 / 651 |

Delta versus benchmark:

| Label | Depth | Runtime delta | Node delta | Node delta % | Simplex iteration delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | -823.14s | -828 | -13.5% | -1101932 |
| 2 | 2 | +273.25s | +695 | +11.3% | +1133883 |
| 2 | 3 | -1219.92s | -2179 | -35.5% | -1762147 |
| 3 | 1 | +3.80s | 0 | 0.0% | +274 |
| 3 | 2 | -0.89s | 0 | 0.0% | -1126 |
| 3 | 3 | -0.12s | 0 | 0.0% | +8161 |
| 7 | 1 | -857.66s | -1726 | -34.6% | -1380802 |
| 7 | 2 | +896.70s | +273 | +5.5% | +929497 |
| 7 | 3 | +371.34s | -441 | -8.8% | +113348 |

Bound-progress threshold times from Gurobi's printed progress table:

| Label | Depth | `BestBd >= -1.0` | `>= -0.75` | `>= -0.5` | `>= -0.25` | `>= -0.1` | `>= 0.0` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 126s | 2280s | 2897s | 3902s | 4321s | never printed |
| 2 | 1 | 175s | 2018s | 2572s | 3206s | 3498s | never printed |
| 2 | 2 | 115s | 641s | 2965s | 4211s | 4583s | never printed |
| 2 | 3 | 136s | 861s | 2572s | 2834s | 3083s | never printed |
| 3 | 0 | 4s | 4s | 14s | 14s | 20s | 27s |
| 3 | 1 | 5s | 5s | 10s | 14s | 28s | 31s |
| 3 | 2 | 4s | 4s | 10s | 10s | 16s | 26s |
| 3 | 3 | 6s | 6s | 12s | 12s | 18s | 27s |
| 7 | 0 | 120s | 120s | 250s | 2377s | 2386s | never printed |
| 7 | 1 | 140s | 1341s | 1412s | never printed | never printed | never printed |
| 7 | 2 | 118s | 127s | 203s | 1977s | 3283s | never printed |
| 7 | 3 | 120s | 126s | 244s | 1820s | 2756s | never printed |

Interpretation:

- Label 2 depth 3 is faster because it explores far fewer nodes than the benchmark, despite each node being somewhat more expensive (`0.782s/node` vs `0.703s/node`). This is a search-tree win, not a per-node cost win.
- Label 2 depth 2 is slower because it explores more nodes and does more simplex work, even though it reaches `BestBd >= -0.75` early. Early bound improvement is not sufficient; the later tree remains hard.
- Label 7 depth 1 is the cleanest speedup: fewer nodes, slightly faster per node, and much fewer simplex iterations.
- Label 7 depth 3 still explores fewer nodes than the benchmark but is slower because each node is substantially more expensive and total simplex work is higher. More intervals can reduce node count but still lose to LP/simplex cost.
- Label 3 is solved at the root node in every depth; splitting does not matter much for it. Runtime variation is mostly root cuts/simplex work and process overhead.

Current conclusion: the log-boolean split encoding is acting as a branching aid. It gives Gurobi extra structured binary variables that can align well with the hard parts of the objective for some labels/depths, reducing the branch-and-bound tree. It is not guaranteed to help because it also increases presolved columns, binary vars, and simplex work per node. The observed runtime is the balance of two forces: fewer prunable branches versus more expensive LP relaxations and a changed branching order.

### 2026-04-25 14:21:07 UTC | Model: Codex GPT-5

Continued looking for deterministic ways to reproduce the faster cases.

Confirmed from the formulation:

- The interval split formulation does not significantly improve the root relaxation for these contiguous exhaustive intervals. The continuous hull is still close to the original `[0, 0.75]` patch range.
- Therefore, the speedup comes from branch-and-bound behavior after Gurobi branches on useful interval/address variables.
- If we want to reproduce the speedup consistently, the right target is not only "add more split depth"; it is "make Gurobi branch on the useful split variables early, without bloating every node too much."

Concrete useful finding #1: singleton interval constraints are redundant.

In `util/ERAN/tf_verify/ai_milp.py:create_model`, the current code enters the log-bool block for every non-fixed input pixel when `add_bool_constraints=True`, even if that pixel has only one interval:

```python
if add_bool_constraints and LB_N0[i] != UB_N0[i]:
    bounds_list = bounds_config[i] if bounds_are_per_pixel else bounds_config
    bounds_len = len(bounds_list)
    num_bits = math.ceil(math.log2(bounds_len))
    bin_vars = model.addVars(num_bits, vtype=GRB.BINARY, ...)
    aux_vars = model.addVars(bounds_len, vtype=GRB.CONTINUOUS, ...)
    model.addConstr(aux_vars.sum() == 1, ...)
    ...
```

For `bounds_len == 1`, this adds one auxiliary selector and three constraints:

- `aux == 1`
- `x <= aux * ub`
- `x >= aux * lb`

Those are redundant because `x` was already created with `lb=LB_N0[i]` and `ub=UB_N0[i]`, and for the base patch bounds those are the same normalized interval. In these runs, depth 0 has 100 patch pixels with one interval, so the benchmark carries roughly 100 redundant aux variables and 300 redundant constraints. Deeper depths still carry singleton constraints for the unsplit patch pixels.

Recommended deterministic code change:

```python
if bounds_len <= 1:
    # Optional safety: tighten var bounds to the one interval if needed.
    # For the current patch setup this should already match LB_N0/UB_N0.
    continue
```

Expected effect: smaller model in every depth, with no semantic change to the verified domain. This will not explain the whole depth speedup, but it should reduce per-node LP cost and model build/presolve cost deterministically.

Concrete useful finding #2: force useful split-bit branching with Gurobi branch priorities.

The fast cases happened when the added split formulation reduced the search tree:

- Label 2 depth 3: `6146 -> 3967` nodes versus benchmark.
- Label 7 depth 1: `4985 -> 3259` nodes versus benchmark.

The slow cases happened when the formulation did not reduce the tree enough or made each node more expensive:

- Label 2 depth 2: `6146 -> 6841` nodes.
- Label 7 depth 3: fewer nodes than benchmark (`4985 -> 4544`) but higher `sec/node`, so total runtime still increased.

This points to a branch-order intervention. When creating `Bit_pixel_i[b]`, set a high `BranchPriority` so Gurobi branches on selected split bits before generic ReLU/maxpool binaries. For bisection intervals, branch the most significant interval bit first, because it separates lower-vs-upper halves of the interval. Within pixels, prioritize higher-gradient pixels first.

Sketch:

```python
priority_base = 1000
pixel_priority = priority_base - gradient_rank
bit_priority = pixel_priority + b  # or prioritize MSB: pixel_priority + num_bits - b
bin_vars[b].BranchPriority = bit_priority
```

This needs `create_model` to receive either split metadata or priority metadata, not just interval lists. The current `bounds` list loses gradient rank/order information. A minimal version could prioritize all `Bit_pixel_*` variables equally above ReLU binaries; a better version should pass the selected pixel order from `choose_split_pixels`.

Why this is promising:

- The root LP is not much stronger, so we should not wait for Gurobi's generic branching heuristic to discover the split bits.
- The split bits represent a human-chosen decomposition of the hard input region. Giving them branch priority should reproduce the tree-shrinking effect more consistently than simply increasing depth.

Concrete useful finding #3: fixed depth is not reliable; use label-sensitive depth or deterministic racing.

Current run:

- Label 2 best depth: 3.
- Label 7 best depth: 1.
- Label 3: root-node solve at all depths, so splitting mostly adds overhead.

Older CSV `csv/recursive_timeout_refinement_20260413_194652_good_toshow_great_improvements_now_why.csv` shows the same non-monotonic pattern across upper bounds:

- `ub=0.8`, label 7 is best at benchmark depth 0, but among split runs depth 3 is better than depth 1 and depth 2. This shows splitting can both help relative to other split depths and still lose to the no-split benchmark.
- `ub=0.9`, label 3 improves from depth 0 about 3096s to depth 3 about 2308s.
- `ub=1.0`, label 3 was best at depth 1 about 7534s, while depth 2/3 were much slower.

Conclusion: there is no globally best depth. A deterministic policy should choose per label and per case.

A practical deterministic strategy:

1. Run the benchmark/all-label attempt as now.
2. For timed-out labels, compute one split-pixel set from the relaxed example.
3. Run short single-label probes for candidate formulations, for example depth 1 and depth 3, with the same deterministic Gurobi settings.
4. Choose the formulation using a score that penalizes both tree size and node cost:

   `score = projected_nodes_to_cutoff * sec_per_node`

   where projected nodes can be estimated from recent `BestBd` movement and unexplored-node growth. Do not use early best-bound alone; label 7 depth 2/3 had better early bounds but worse final runtime.

This "racing" costs extra time, but for hour-scale solves it can prevent selecting a disastrous depth. For shorter solves it may not be worth it.

Concrete useful finding #4: split pixel sets overlap heavily, so a smaller/global split set may preserve benefit with less overhead.

Depth-1 selected pixels from the current logs:

- Label 2: 30 pixels.
- Label 3: 30 pixels.
- Label 7: 30 pixels.
- Pairwise overlaps: label 2/3 = 25, label 2/7 = 24, label 3/7 = 27.
- Intersection across labels 2/3/7 = 24 pixels.
- Union across labels 2/3/7 = 38 pixels.

The common 24 pixels are:

`[197, 198, 199, 200, 201, 202, 203, 204, 205, 225, 226, 227, 228, 229, 230, 231, 232, 233, 254, 256, 257, 258, 260, 261]`

This suggests two experiments:

- Split only the common/core 24 pixels first, with high branch priority. This may keep most of the branch-guidance benefit while reducing per-node cost.
- Build one shared split model for all timed-out labels using the 38-pixel union, then reuse the model with different objectives. This mirrors the benchmark's model-reuse behavior, but adds a controlled split formulation. It may be better than building three separate per-label models with slightly different split sets.

Recommended next experiments, in order:

1. Patch `create_model` to skip `bounds_len <= 1` singleton selector constraints. This should be semantics-preserving and deterministic.
2. Add optional `BranchPriority` for `Bit_pixel_*` variables, first with a uniform high priority above network binaries.
3. Add logging for selected split pixel order and for count of split-bit variables, singleton intervals skipped, and priority settings.
4. Test four controlled schedules on the same `20260424_134749` setup:
   - current formulation,
   - skip-singleton only,
   - skip-singleton + split-bit branch priority,
   - skip-singleton + branch priority + core-24 split set.
5. Only after this, try adaptive racing; branch priority is simpler and more directly tied to the observed reason for the speedup.

### 2026-05-06 18:09:23 UTC | Model: Codex GPT-5

Implemented the deterministic formulation controls recommended above. Files changed:

- `util/ERAN/tf_verify/ai_milp.py`
- `util/ERAN/tf_verify/config.py`
- `util/ERAN/tf_verify/__main__.py`
- `run_verifier.py`
- `utils.py`
- `scripts/run_multiple_till_sol.py`
- `scripts/run_split_priority_experiments.py`

Implementation details:

- `create_model` now has configurable singleton skipping via `config.skip_singleton_bounds`.
- Default is `skip_singleton_bounds=True`, so singleton interval selectors are skipped in normal patched runs.
- The old/current formulation remains reproducible with `--skip_singleton_bounds False`; this is needed for the controlled baseline.
- For singleton intervals, the code tightens the input variable bounds to the singleton interval before continuing. This makes the skip semantics-preserving even if a future singleton is narrower than the original input-box bound.
- `create_model` now has configurable split-bit branch priority via `config.split_bit_branch_priority`.
- Default is `0`, which disables priority and leaves Gurobi's default branching behavior.
- When `split_bit_branch_priority > 0`, every created `Bit_pixel_i[b]` variable gets that uniform `BranchPriority` value. The first experiment uses priority `1000`, above ordinary network binaries whose default priority is `0`.

Logging added:

- ERAN/MILP logs now print:
  - `skip_singleton_bounds`
  - `split_bit_branch_priority`
  - count of selector pixels
  - count of split-bit variables
  - count of auxiliary selector variables
  - count of singleton intervals skipped
  - branch-priority value
  - count of variables that received priority
- `run_verifier.py` prints selected split pixel order before each recursive rerun.
- Detailed CSV rows now include:
  - `skip_singleton_bounds`
  - `split_bit_branch_priority`
  - `split_indices`

Controlled experiment runner:

- Added `scripts/run_split_priority_experiments.py`.
- It uses the same core setup as `20260424_134749`: image index `2`, patch `(0, 0)`, patch size `10`, upper bound `0.75`, max depth `4`, top-k `30`.
- It runs the original four timeout schedules:
  - `[720000]`
  - `[10, 720000]`
  - `[10, 10, 720000]`
  - `[10, 10, 10, 720000]`
- It runs these four formulation variants:
  - `current_formulation`: `skip_singleton_bounds=False`, `split_bit_branch_priority=0`
  - `skip_singleton_only`: `skip_singleton_bounds=True`, `split_bit_branch_priority=0`
  - `skip_singleton_branch_priority`: `skip_singleton_bounds=True`, `split_bit_branch_priority=1000`
  - `skip_singleton_branch_priority_core24`: `skip_singleton_bounds=True`, `split_bit_branch_priority=1000`, fixed split set `[197, 198, 199, 200, 201, 202, 203, 204, 205, 225, 226, 227, 228, 229, 230, 231, 232, 233, 254, 256, 257, 258, 260, 261]`

Verification performed:

- Ran syntax compilation:
  - `python3 -m py_compile run_verifier.py utils.py scripts/run_multiple_till_sol.py scripts/run_split_priority_experiments.py util/ERAN/tf_verify/ai_milp.py util/ERAN/tf_verify/__main__.py util/ERAN/tf_verify/config.py`
- Ran lightweight import/config checks:
  - imported `scripts.run_split_priority_experiments`
  - confirmed 4 variants and 4 timeout schedules are configured
  - confirmed the core-24 fixed split list is wired into the last variant

What was not run:

- I did not launch the full controlled experiment because the configured schedules include `720000` second MILP timeouts and can run for many hours. The next agent should intentionally launch `python3 scripts/run_split_priority_experiments.py` when long-running experiments are desired.

Current conclusion:

- The implementation work for singleton skipping, uniform split-bit branch priority, split-order logging, model-stat logging, and controlled experiment setup is complete at the code level.
- There is not yet new runtime evidence showing whether branch priority improves the `20260424_134749` case. That evidence requires running the controlled experiment script and comparing the resulting CSV/logs.

Continuation notes for the next agent:

- First inspect `git status --short`; at this point there should be six modified tracked files, plus the new `scripts/run_split_priority_experiments.py` and this `journal/` directory may appear as untracked if the local repo has not added them.
- To run the full experiment, execute `python3 scripts/run_split_priority_experiments.py` from the repo root.
- The detailed CSV path is printed at startup and uses the `recursive_timeout_refinement_<timestamp>_split_priority.csv` naming pattern through `resolve_runs_csv_path`.
- The most important log lines to compare are `Input split selector stats`, Gurobi model-size/presolve summaries, final per-label statuses, explored nodes, simplex iterations, and total solver/runtime per adversarial label.

### 2026-05-06 18:13:26 UTC | Model: Codex GPT-5

Correction/update: split-bit branch priority is now controlled by an explicit on/off flag, not by using `0` as an implicit disable value.

Updated behavior:

- `enable_split_bit_branch_priority=False` means no `BranchPriority` is applied to `Bit_pixel_*` variables.
- `enable_split_bit_branch_priority=True` means each created `Bit_pixel_i[b]` gets `BranchPriority = split_bit_branch_priority`.
- `split_bit_branch_priority` is now the numeric priority value only. It defaults to `1000`.
- The model validates that the numeric priority is positive only when the flag is enabled.

Files updated in this correction:

- `util/ERAN/tf_verify/config.py`
- `util/ERAN/tf_verify/__main__.py`
- `util/ERAN/tf_verify/ai_milp.py`
- `run_verifier.py`
- `utils.py`
- `scripts/run_multiple_till_sol.py`
- `scripts/run_split_priority_experiments.py`

Controlled experiment variant settings are now:

- `current_formulation`: `skip_singleton_bounds=False`, `enable_split_bit_branch_priority=False`, `split_bit_branch_priority=1000`
- `skip_singleton_only`: `skip_singleton_bounds=True`, `enable_split_bit_branch_priority=False`, `split_bit_branch_priority=1000`
- `skip_singleton_branch_priority`: `skip_singleton_bounds=True`, `enable_split_bit_branch_priority=True`, `split_bit_branch_priority=1000`
- `skip_singleton_branch_priority_core24`: `skip_singleton_bounds=True`, `enable_split_bit_branch_priority=True`, `split_bit_branch_priority=1000`, fixed core-24 split set

Additional logging/CSV fields:

- Logs now print `enable_split_bit_branch_priority`.
- The selector stats line now prints `branch_priority_enabled`.
- Detailed CSV rows now include `enable_split_bit_branch_priority`.

Verification after this correction:

- Re-ran `python3 -m py_compile run_verifier.py utils.py scripts/run_multiple_till_sol.py scripts/run_split_priority_experiments.py util/ERAN/tf_verify/ai_milp.py util/ERAN/tf_verify/__main__.py util/ERAN/tf_verify/config.py`.
- Re-imported `scripts.run_split_priority_experiments` and confirmed the four variants are `[False, False, True, True]` for `enable_split_bit_branch_priority`, all with value `1000`.

Current conclusion:

- The branch-priority experiment now has a real boolean switch. The value `1000` is just the priority level used when the switch is on.

### 2026-05-06 18:17:41 UTC | Model: Codex GPT-5

Added next-step comparison task for testing the explicit branch-priority flag.

Recommended quick comparison:

1. In `scripts/run_multiple_till_sol.py`, use the same image, patch, upper bound, max depth, top-k, and timeout schedule for both runs.
2. First run with branch priority off:

   ```python
   SKIP_SINGLETON_BOUNDS = True
   ENABLE_SPLIT_BIT_BRANCH_PRIORITY = False
   SPLIT_BIT_BRANCH_PRIORITY = 1000
   FIXED_SPLIT_INDICES = None
   ```

3. Then run with branch priority on:

   ```python
   SKIP_SINGLETON_BOUNDS = True
   ENABLE_SPLIT_BIT_BRANCH_PRIORITY = True
   SPLIT_BIT_BRANCH_PRIORITY = 1000
   FIXED_SPLIT_INDICES = None
   ```

4. For a smoke test, temporarily shrink `RUN_SCHEDULE` to one short schedule such as:

   ```python
   RUN_SCHEDULE = [
       {"upper_bound": 0.75, "timeout_milps": [10, 10]},
   ]
   ```

5. For the full controlled experiment, run:

   ```bash
   python3 scripts/run_split_priority_experiments.py
   ```

   Warning: the full script includes `720000` second MILP timeouts and can run for many hours.

What to compare:

- In ERAN logs, confirm the flag was applied:
  - off run should show `enable_split_bit_branch_priority=False`, `branch_priority_enabled=False`, and `branch_priority_vars=0`.
  - on run should show `enable_split_bit_branch_priority=True`, `branch_priority_enabled=True`, and `branch_priority_vars > 0` once split-bit variables exist.
- In the detailed CSV, compare rows with the same `upper_bound`, `timeout_milp_s`, `adv_label`, and `current_depth`.
- Main metrics:
  - `attempt_runtime_seconds`
  - `iteration_runtime_seconds`
  - `status_name`
  - `final_outcome`
  - `log_path`
- In Gurobi log text for the matching `log_path`, compare:
  - explored nodes
  - simplex iterations
  - solver runtime
  - presolved columns and binary variables
  - bound-progress timing from the printed progress table

Conclusion criteria:

- Branch priority is useful if the on run solves the same label/depth with fewer nodes or lower runtime without increasing per-node cost enough to erase the benefit.
- If runtime gets worse while node count decreases, the priority may still be reducing the tree but making each node more expensive; compare simplex iterations and seconds per node before concluding it is bad.
- If `branch_priority_vars=0`, the run did not actually create prioritized split-bit variables, so that attempt is not a valid branch-priority test.

### 2026-05-08 07:04:34 UTC | Model: Codex GPT-5

Inspected the active priority-enabled logs under:

- `logs/20260506/20260506_181950_run_02_ub_0p75_timeouts_10_720000`

Provisional finding: the priority-enabled depth-1 rerun for adversarial label 2 is dramatically slower than the old April depth-1 result and was still running at inspection time.

Evidence from the active run:

- `001_adv_label_2_eran_run.log` has `enable_split_bit_branch_priority=True`, `split_bit_branch_priority=1000`.
- The split-bit priority was actually applied: `branch_priority_enabled=True` and `branch_priority_vars=60`.
- The single-label rerun had reached about `125520s` in the Gurobi progress table and was still not complete.
- At that point it had explored about `189876` nodes and still had best bound around `-1.01326`.
- The subprocess was still active as PID `350761` with elapsed time around `125671s`.

Comparison points:

- The old `20260424_134749` depth-1 label-2 rerun finished around `3498s` solver time / `3511.873s` wall time.
- The May 6 priority-enabled depth-1 run is already more than `35x` slower than that old depth-1 result and still not finished.
- The May 6 all-label no-split run had `branch_priority_vars=0`, so branch priority did not affect that initial model. Its label-2 runtime was `3868.3567s`, which is not worse than the old no-split label-2 runtime of about `4321s`; this suggests the severe regression appears when split-bit priorities are active, not from singleton skipping alone.

Important caveat:

- This is not a clean off-vs-on controlled comparison because the May 6 batch appears to have priority enabled for the run, and the matching priority-off depth-1 rerun has not been run side-by-side under the exact same code/settings.
- The current evidence is still strong enough to say the uniform high branch priority is behaving badly for label 2 in this run.

Current conclusion:

- The user is likely correct: with uniform high priority on split-bit variables, label 2 at depth 1 is taking much longer.
- The next controlled test should stop relying on this long run as a success candidate and run a direct priority-off vs priority-on comparison with the same `skip_singleton_bounds=True` setup.
- If using branch priority again, consider reducing the priority, prioritizing only a small/core split set, or assigning priority to fewer bits instead of all split-bit variables uniformly.
