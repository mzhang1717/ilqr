# iLQR Trajectory Optimization

This project implements an iterative Linear Quadratic Regulator (iLQR) pipeline for a kinematic bicycle model with trajectory-tracking costs and control limits.

## Project Summary

The codebase provides:

- A **kinematic bicycle dynamics model** (`BicycleModel`)
- A **trajectory representation and spline utilities** (`Trajectory`)
- A **tracking cost function** with cross-track, heading, speed, and control penalties (`BicycleTrackingCost`)
- An **iLQR solver** (`ILQR`) with:
  - backward pass
  - constrained control update (Box-DDP style projected Newton)
  - forward pass + Armijo backtracking line search
  - solve output export/printing
- A **rollout controller** (`RolloutController`) used to generate the initial rollout trajectory

## Main Components

- `motion_model.h`: base motion model interface + control bound storage
- `bicycle_model.h/.cpp`: bicycle dynamics, Jacobians, control limits, JSON config loading
- `trajectory.h/.cpp`: trajectory storage, spline fitting, tracking errors, Jacobians
- `cost_function.h`: base cost function interface
- `bicycle_tracking_cost.h/.cpp`: tracking objective + derivatives + JSON weight loading
- `rollout_controller.h/.cpp`: initial-guess rollout generation for bicycle tracking
- `ilqr.h/.cpp`: iLQR solver, configuration loading, solve output utilities

## Configuration

Runtime parameters are read from `ilqr_config.json`.

Supported sections:

- `bicycle_model`
  - `wheel_base`
  - `control_lower_bound`
  - `control_upper_bound`
- `ilqr_solver`
  - `horizon`
  - `step_interval`
  - `max_iterations`
  - `converge_tolerance`
  - `learning_rate`
  - `line_search`
    - `armijo_coefficient`
    - `backtracking_decay`
    - `minimum_step`
  - `regularization`
    - `regularization_scale`
    - `max_projected_newton_iterations`
    - `active_set_tolerance`
    - `step_tolerance`

For cost weights, `BicycleTrackingCost` reads from section:

- `bicycle_tracking_cost`

## Build and Test

Build:

```bash
bazel build //:main
```

Run tests:

```bash
bazel test //:trajectory_test //:bicycle_model_test //:bicycle_tracking_cost_test //:ilqr_test
```

## Solve Output Utilities

`ILQR` can output trajectories after solve:

- `get_solve_output()`
- `print_solve_output()`
- `save_solve_output_csv(path_prefix)`

CSV files are generated for reference, initial rollout, and final optimized trajectories (states and controls).

## Visualization

Use the helper script:

```bash
python3 plot_solve_output.py --prefix /tmp/ilqr_bicycle_test
```

Optionally save a figure:

```bash
python3 plot_solve_output.py --prefix /tmp/ilqr_bicycle_test --save result.png
```

## Current Status

The project includes unit tests for major components and supports JSON-driven configuration for model, cost, and solver parameters.
