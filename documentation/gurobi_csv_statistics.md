# Gurobi CSV Statistics

`run_multiple_till_sol.py` writes detailed per-label rows through `run_verifier.py`.
Those rows now include Gurobi statistics parsed from each adversarial-label log block.

## PrintStats Columns

`model.printStats()` output is parsed only when it is inside the current analyzer markers:

- `START printStats Regular adv_label=<label>` through `END printStats Regular adv_label=<label>`
- `START printStats Presolved adv_label=<label>` through `END printStats Presolved adv_label=<label>`

Older labels such as `Original model Print stats` and `Presolved model Print stats` are not used for printStats parsing.

Regular printStats columns keep their base names. Presolved printStats columns add `_presolved`, for example:

- `linear_constraint_matrix_constrs`
- `linear_constraint_matrix_vars`
- `linear_constraint_matrix_nzs`
- `variable_types_continuous`
- `variable_types_integer`
- `variable_types_binary`
- `matrix_coefficient_range`
- `variable_bound_range`
- `linear_constraint_matrix_constrs_presolved`
- `variable_types_binary_presolved`
- `rhs_coefficient_range_presolved`

Range values are stored as printed inside the log, including bracket notation.
Missing values are blank CSV cells.

## Solve Columns

The same per-label block is also parsed for solve-level Gurobi output:

- `optimizer_presolve_removed_rows`, `optimizer_presolve_removed_columns`, `optimizer_presolve_time_seconds`
- `optimizer_presolved_rows`, `optimizer_presolved_columns`, `optimizer_presolved_nonzeros`
- `optimizer_presolved_variable_types_continuous`, `optimizer_presolved_variable_types_integer`, `optimizer_presolved_variable_types_binary`
- `extra_simplex_iterations_after_uncrush`, `extra_simplex_iterations_dual_to_original`
- `root_relaxation_result`, `root_relaxation_objective`, `root_relaxation_iterations`, `root_relaxation_seconds`
- `explored_nodes`, `explored_simplex_iterations`, `explored_time_seconds`
- `mip_final_status`, `mip_final_runtime_seconds`, `mip_final_nodes`, `mip_final_objbound`, `mip_final_solcnt`

Cutting plane names are dynamic columns using `cutting_planes_<normalized name>`, such as
`cutting_planes_cover` or `cutting_planes_relax_and_lift`.

## Relaxed Fallback Columns

When a label block contains `No solution found, running relaxed model`, relaxed-model statistics are written with a `relaxed_` prefix. Examples:

- `relaxed_presolve_removed_steps`
- `relaxed_presolve_time_seconds`
- `relaxed_presolved_rows`, `relaxed_presolved_columns`, `relaxed_presolved_nonzeros`
- `relaxed_barrier_dense_cols`, `relaxed_barrier_free_vars`, `relaxed_barrier_threads`
- `relaxed_solved_iterations`, `relaxed_solved_seconds`, `relaxed_optimal_objective`

If Gurobi reports multiple relaxed presolve removal lines, they are stored together in `relaxed_presolve_removed_steps`.

## CSV Header Behavior

The original recursive timeout refinement columns remain first. Known Gurobi statistic columns follow them.
If a new cutting-plane or barrier-stat name appears later, the CSV header is extended and earlier rows are kept with blank values for the new column.
