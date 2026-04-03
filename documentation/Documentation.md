## Initial run
**File:**
 -  patch_attack_verification_log_initial_run_find_which_is_adversarial_and_timeout.csv

This is a csv representing a run of multiple examples, with a patch in the corner (0,0) with varying patch sizes going from 3 to 12 on 10 different indexes
with a timeout of 30 minutes. This helped us understand which examples were robust or not, if we were proving or not their robustness, if we were getting a time out on them.



### Implementation of Boolean Values for multiple bounds as constraints:
**File:**
 - middle_bounds_first_test_trivial_bool_encoding_2_intervals.csv

**Code:**
```python
	# OLD IMPLEMENTATION WITH TWO INTERVALS ONLY
	bool_var_name = "I"+str(i)
	bool_var = model.addVar(vtype=GRB.BINARY, name=bool_var_name)
	bool_var_list.append(bool_var)
	# TODO change later to middle bound = average or something else 

	# if I = 0 then upper part else lower part  
	# x <= ub - (ub - mb)*I
	expr = var - UB_N0[i] + (UB_N0[i] - middle_bound) * bool_var
	model.addConstr(expr, GRB.LESS_EQUAL, 0)
	# x >= mb - (mb - lb)*I
	expr = var - middle_bound + (middle_bound - LB_N0[i]) * bool_var
	model.addConstr(expr, GRB.GREATER_EQUAL, 0)

```

First run with 2 intervals, with one boolean value encoding


**File:**
 - multiple_intervals_milp_test_multiple_bounds_implementation_first_run_trivial_multiple_choice_in_input_box.csv

Run with multiple bounds using trivial boolean values for any number of intervals, meaning for for each bound we had a bool for it and we on top of it added a constraint that the sum of the boolean values are 1.
**Constraints:**
 - x <= Bool1 * UpperBound1 + Bool2 * UpperBound2 + ...
 - x >= Bool1 * LowerBound1 + Bool2 * LowerBound2 + ...
 - sum(Bool_i) = 1


**Files:**
 
 - multiple_intervals_milp_test_14_01_26_trivial_bool_encoding.csv 
 - multiple_intervals_milp_test_log_bool_values_2026_01_16_17_22_12.csv

This csv "multiple_intervals_milp_test_log_bool_values_2026_01_16_17_22_12.csv" compares trivial boolean encoding where we have N Boolean values for N elements in our bounds list (bounds list - represent the partition of the pixels upper and lower bound) to logarithmic boolean encoding where we have N continuous variables and log(N) boolean variables.
We saw that log boolean encoding improved the time for the run compared to the trivial encoding, But compared to a regular run with a single bounds [0, Upperbound] improvement wasnt assured, and was mostly dependent on how the bounds were chosen and how many bounds there were.


## Next Step 
After timeout, we solve relaxed model to get continuous solution to problem which "was already" solved by the integer model
From relaxed example, we derived which are the pixels in the patch that have the biggest gradient on the function: 
```python
    z_correct = outputs[0, correct_class_idx]
    z_target  = outputs[0, target_class_idx]
    
    margin = z_correct - z_target
```
example in derive_bounds_test jupyter notebook
next we do unnormalize to get the closest value that caused this big gradient and we then split the top30 pixels (ordered by gradients) at the value of the relaxed example 
and rerun 
need to do the infra to rerun all this with new bounds, preferably all inside eran and not rerun everything.

## 11/03/2026 - Come Back after holiday

We don't think we need to do unnormalize to the relaxed example we extracted, because in the end we normalize pictures to pass them to the model.
So assuming our example is already in normalized values, we keep it as is and derive the important pixels from it.

Current test: we want to replace the bounds param from the config which represent the one bound that will be the same for all pixels, to a bounds list per pixel
so that each pixel can get different bounds.

21/03/2026
First try, by hand without a pipeline to check if the splitting solution based on a relaxed adversarial example is viable and promising.
Results:
	- 31.49s for verifying Label 2 without split
	- 25.83 for verifiying Label 2 after first split on 30 pixels
	- 25.63 for verifiying Label 2 after second split on 30 pixels (same pixels to split)

## Recursive timeout refinement flow

The current implementation runs one full ERAN verification pass first and uses that pass only to identify which adversarial labels prove robust quickly, which are adversarial, and which labels timed out.

For each timed-out label, the script starts a separate refinement loop:

1. Take the relaxed example associated with that specific label from the ERAN log.
2. Compute the gradient of the margin `z_correct - z_target` with respect to the input pixels.
3. Restrict attention to the chosen patch and rank patch pixels by gradient magnitude.
4. Select the top `k` patch pixels and split their current bounds into smaller intervals (by half).
5. Rerun ERAN for that single adversarial label only, using the refined per-pixel bounds.
6. If the label is now verified or adversarial, stop for that label.
7. If the label still times out, repeat the same process until `max_depth` is reached.


## 03/04/26

The pipeline to run the labeled that got a timeout with a split derived from the relaxed adversarial example is fully functionable with a split on 30 pixels.
We got a csv that details the whole run `/root/Projects/Nathan/Patch-Attack-Verification/csv/recursive_timeout_refinement_20260326_175035_TODO.csv` and a csv detailing the same csv for each label run so that it will be easier in the future to compare to our benchmarks, here: `/root/Projects/Nathan/Patch-Attack-Verification/csv/recursive_timeout_refinement_20260326_175035_per_label_TODO.csv`.

We are currently running the same upper bounds with a maximal timeout of 72k seconds (similar to without a timeout) just to get the end result.
After this run ends, at our next meeting, our goal is to compare this run to the previous runs with the splits and check if a specific split of the run got better times and by how much did we improve run time for this specific label. We would like also to create a few graphs that display the comparison of all the different runs up to now, with the different solutions we came with from the start, that also includes the last runs with the splits, and show it as bar plot.

Then after this, we ll do a meeting with Dana and check which direction should we continue, we understand that there is multiple degrees of freedom such as: 
 - the number of pixels the split is done on and how do we split
 - how should we choose the timeout (dynamic timeout, maybe based on solutions of relaxed)
 - run from the start with one or multiple splits to get a shortcut to the solution and the run time and how do we choose the number of splits

REMARK: We're not sure if stopping the splits as soon as we got a solution is the right thing and maybe we should check up to a maximum number of splits to maybe define an improvent in times at a specific split, regardless of whether it solved earlier or not.