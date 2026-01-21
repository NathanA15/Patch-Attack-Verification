# Logarithmic Formulation for Interval Selection in Gurobi

This document explains the optimization technique used to select one active interval from a list of $N$ possibilities using **$\lceil \log_2 N \rceil$ binary variables** instead of the standard $N$ binary variables.

## 1. The Concept

In Mixed-Integer Programming (MIP), the complexity of the problem is heavily driven by the number of **Binary (Integer) Variables**. Each binary variable potentially doubles the size of the Branch-and-Bound search tree.

**The Goal:**
Select one set of bounds $[LB_i, UB_i]$ from a list of $N$ possible bound configurations for a variable $x$.

**The Approaches:**
1.  **Standard (One-Hot):** Create $N$ binary variables. $b_i = 1$ if interval $i$ is selected.
2.  **Logarithmic (Vielma-Nemhauser):** Create $\approx \log_2 N$ binary variables that represent the *index* of the selected interval in binary code.

## 2. Why is this Better?

We trade "expensive" binary variables for "cheap" continuous variables.

| Feature | Standard (One-Hot) | Logarithmic (Proposed) |
| :--- | :--- | :--- |
| **Variables** | $N$ Binary | $\lceil \log_2 N \rceil$ Binary + $N$ Continuous |
| **Search Space** | $\approx 2^N$ branches | $\approx 2^{\log_2 N} = N$ branches |
| **Solver Effort** | High (Exponential growth) | Low (Linear growth) |
| **Scalability** | Poor for large $N$ | Excellent for large $N$ |

**Example ($N=8$ intervals):**
* **Standard:** 8 Binary variables ($2^8 = 256$ worst-case nodes).
* **Logarithmic:** 3 Binary variables ($2^3 = 8$ worst-case nodes).

*Note: Continuous variables are solved via Linear Programming (Simplex/Barrier), which is extremely fast compared to Integer branching.*

## 3. Mathematical Logic (How it works)

We use two sets of variables:
1.  **Binary Variables ($z$):** Act as the **Address**. They hold the binary representation of the selected index.
2.  **Continuous Aux Variables ($\lambda$):** Act as the **Selectors**. There is one $\lambda_i$ for each interval.

### The Constraints
1.  **Unity Sum:** The sum of all selectors must be 1.
    $$\sum \lambda_i = 1$$
2.  **Linkage (The Decoder):** We link the bits of the binary address to the selectors.
    * For every bit position $k$, the sum of $\lambda_i$ (where index $i$ has bit $k$ set to 1) must equal the binary variable $z_k$.

### Trace Example: Selecting Index 2 (Binary `10`) from 4 options
Let $N=4$ (Indices 0, 1, 2, 3). We need 2 bits ($z_0, z_1$).
The solver chooses Index 2, so it sets **$z_1=1, z_0=0$**.

**Constraint Check:**
1.  **Bit 0 Link (LSB):** Indices with bit 0 set are 1 (`01`) and 3 (`11`).
    $$\lambda_1 + \lambda_3 = z_0 \Rightarrow \lambda_1 + \lambda_3 = 0$$
    *Result:* $\lambda_1=0, \lambda_3=0$ (since variables are non-negative).

2.  **Bit 1 Link (MSB):** Indices with bit 1 set are 2 (`10`) and 3 (`11`).
    $$\lambda_2 + \lambda_3 = z_1 \Rightarrow \lambda_2 + \lambda_3 = 1$$
    *Since $\lambda_3=0$ (from step 1), result:* $\lambda_2=1$.

3.  **Unity Sum:**
    $$\lambda_0 + \lambda_1 + \lambda_2 + \lambda_3 = 1$$
    $$\lambda_0 + 0 + 1 + 0 = 1 \Rightarrow \lambda_0 = 0$$

**Final Result:** $\lambda = [0, 0, 1, 0]$.
The bounds are applied as:
$$x \le \sum (\lambda_i \cdot UB_i) = 1 \cdot UB_2$$

## 4. Implementation Code

```python
import math
from gurobipy import GRB

# ... inside your pixel loop ...

if add_bool_constraints and LB_N0[i] != UB_N0[i]:
    bounds_list = config_param.bounds
    bounds_len = len(bounds_list)
    
    # 1. Calculate required bits
    num_bits = math.ceil(math.log2(bounds_len))
    
    # 2. Binary Variables ("The Address")
    bin_vars = model.addVars(num_bits, vtype=GRB.BINARY, name=f"Bit_{i}")
    
    # 3. Auxiliary Continuous Variables ("The Selectors")
    aux_vars = model.addVars(bounds_len, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"Aux_{i}")

    # 4. Unity Constraint (Only one aux var can be active)
    model.addConstr(aux_vars.sum() == 1, name=f"Sum_Aux_{i}")
    
    # 5. Linkage Constraints (Decoder Logic)
    for b in range(num_bits):
        # Find indices where the b-th bit is 1
        indices_with_bit_set = [j for j in range(bounds_len) if (j >> b) & 1]
        
        if indices_with_bit_set:
            model.addConstr(
                sum(aux_vars[j] for j in indices_with_bit_set) == bin_vars[b],
                name=f"Link_Bit_{i}_{b}"
            )

    # 6. Apply Bounds using Convex Combination
    model.addConstr(
        var <= sum(aux_vars[j] * bounds_list[j][1] for j in range(bounds_len)),
        name=f"UB_Constr_{i}"
    )
    model.addConstr(
        var >= sum(aux_vars[j] * bounds_list[j][0] for j in range(bounds_len)),
        name=f"LB_Constr_{i}"
    )



