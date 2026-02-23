#pragma once

#include <eigen3/Eigen/Dense>

#include "bicycle_model.h"
#include "trajectory.h"

class RolloutController {
public:
    RolloutController(
        BicycleModel& model,
        const Trajectory& reference,
        double k_cte = 0.6,
        double k_heading = 1.2,
        double k_speed = 0.8);

    Trajectory generate_initial_trajectory(const Eigen::VectorXd& x0, int horizon) const;

private:
    int closest_reference_index_(const Eigen::VectorXd& x) const;
    double reference_curvature_(int idx) const;

    BicycleModel* model_;
    const Trajectory* reference_;
    double k_cte_;
    double k_heading_;
    double k_speed_;
};
