# Full-Coeff AxSDP Design

This repository contains the current Julia implementation of the AxSDP design workflow, including the `full_coeff` algorithm-design model.

The `full_coeff` pipeline solves three stages in sequence:

1. a fixed-parameter dual SDP warm-start problem,
2. a local nonconvex QCQP design problem with Ipopt,
3. a global nonconvex QCQP design problem with Gurobi.

The main entry point is:

```bash
axsdp-joint/run_general_design_experiment.jl
```

## Core files for `full_coeff`

If you want to keep only the minimum runnable subset for the `full_coeff` mode, these files are required:

- `Project.toml`
- `Manifest.toml`
- `axsdp-joint/run_general_design_experiment.jl`
- `axsdp-joint/BnB_PEP_axsdp_joint_design_full_coefficients.jl`
- `axsdp-joint/BnB_PEP_axsdp_joint_design_general_helpers.jl`
- `axsdp-joint/BnB_PEP_axsdp_joint_design.jl`
- `axsdp-joint/BnB_PEP_axsdp_joint_interpolation.jl`
- `function-value/code_to_compute_pivoted_cholesky.jl`

The dependency chain is:

```text
run_general_design_experiment.jl
  -> BnB_PEP_axsdp_joint_design_full_coefficients.jl
  -> BnB_PEP_axsdp_joint_design_general_helpers.jl
  -> BnB_PEP_axsdp_joint_design.jl
  -> BnB_PEP_axsdp_joint_interpolation.jl
  -> function-value/code_to_compute_pivoted_cholesky.jl
```

## Solver stack

The checked-in `Project.toml` includes the main Julia dependencies:

- `JuMP`
- `Ipopt`
- `Gurobi`
- `Clarabel`

Optional:

- `MosekTools` and `Mosek` for fixed SDP solves

Solver usage in the current workflow:

- fixed dual SDP: `Mosek` if available, otherwise `Clarabel`
- local design QCQP: `Ipopt`
- global design QCQP: `Gurobi`

The global `full_coeff` solve needs a working Gurobi installation and license.

## Environment setup

Install Julia, then instantiate the checked-in environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

If you are running in a fresh clone, do this from the repository root.

## Local usage

The direct Julia interface is:

```bash
julia --project=. axsdp-joint/run_general_design_experiment.jl <design_mode> <N> <qcqp_time_limit_sec> [run_dir]
```

For the `full_coeff` mode, a typical local command is:

```bash
julia --project=. axsdp-joint/run_general_design_experiment.jl full_coeff 3 300
```

Another example with a custom output directory:

```bash
julia --project=. axsdp-joint/run_general_design_experiment.jl \
  full_coeff 4 18000 ./outputs/manual_full_coeff_N4
```

If `run_dir` is omitted, the script writes to a timestamped folder under:

```text
outputs/axsdp_design_runs/
```

## What the driver does

For `full_coeff`, the driver does the following:

1. builds a fixed ALM-like baseline instance,
2. solves the fixed dual SDP,
3. builds the `full_coeff` design instance,
4. extracts warm-start data and PSD/factor bounds from the fixed SDP solution,
5. solves a local design QCQP,
6. uses the local solution to warm-start the global Gurobi QCQP,
7. writes checkpoints and summaries after each stage.

This means the first stage is always a fixed-parameter dual SDP, even when the final design mode is `full_coeff`.

## Important environment variables

`run_general_design_experiment.jl` reads the following environment variables:

- `AXSDP_DESIGN_MODE`
- `AXSDP_RESUME`
- `AXSDP_ALLOW_WEIGHT_ON_X0`
- `AXSDP_ADD_PSD_CUTS`
- `AXSDP_SHOW_SOLVER_OUTPUT`
- `AXSDP_EQUALITY_TOL`
- `AXSDP_LOCAL_MAX_ITER`
- `AXSDP_LOCAL_TOL`
- `AXSDP_LOCAL_ACCEPTABLE_TOL`
- `AXSDP_GUROBI_THREADS`
- `AXSDP_GUROBI_MIPFOCUS`
- `AXSDP_RHO_DUAL_UPPER`
- `AXSDP_SMOOTHNESS_L`
- `AXSDP_MU_A`
- `AXSDP_L_A`
- `AXSDP_RHO_DUAL0`
- `AXSDP_ETA0`
- `AXSDP_RX2`
- `AXSDP_RY2`

Example:

```bash
AXSDP_DESIGN_MODE=full_coeff \
AXSDP_GUROBI_THREADS=4 \
AXSDP_SHOW_SOLVER_OUTPUT=1 \
julia --project=. axsdp-joint/run_general_design_experiment.jl full_coeff 5 21600
```

## Output files

Each run directory contains stage checkpoints and text summaries such as:

- `fixed_sdp_solution.bin`
- `local_design_solution.bin`
- `global_design_solution.bin`
- `run_state.txt`
- `summary.txt`
- `run_config.txt`

When using the shell wrapper on Palmetto, additional logs are written, including:

- `driver_metadata.txt`
- `launch_command.txt`
- `julia.stdout.log`

## Palmetto usage

For Palmetto runs, use:

- `palmetto/run_axsdp_general_design.slurm`
- `scripts/run_axsdp_general_design_job.sh`

Typical submission:

```bash
sbatch \
  --export=ALL,REPO_DIR="$HOME/AGD",JULIA_BIN="$JULIA_BIN",GUROBI_MODULE="$GUROBI_MODULE",AXSDP_DESIGN_MODE=full_coeff,N=3,QCQP_TIME_LIMIT=18000 \
  palmetto/run_axsdp_general_design.slurm
```

The shared runtime wrapper will create a timestamped run directory under:

```text
$REPO_DIR/outputs/palmetto_general_runs/
```

## Notes

- `AXSDP_ETA0` is used when building the design-side initialization, but the fixed SDP stage still uses the default ALM-like fixed instance construction unless the driver code is changed.
- `mu_A` is treated as a nonnegative lower-spectrum parameter in the operator interpolation model. The implementation allows zero eigenvalues.
- The repository also contains other design modes (`single_tau`, `multi_tau`, `abcd`), but the file list above is the minimum subset for `full_coeff`.
