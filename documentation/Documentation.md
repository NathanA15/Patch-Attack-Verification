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

