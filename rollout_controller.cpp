#include "rollout_controller.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
double wrap_angle(double angle) {
    constexpr double kPi = 3.14159265358979323846;
    while (angle > kPi) {
        angle -= 2.0 * kPi;
    }
    while (angle < -kPi) {
        angle += 2.0 * kPi;
    }
    return angle;
}
}  // namespace

RolloutController::RolloutController(
    BicycleModel& model,
    const Trajectory& reference,
    double k_cte,
    double k_heading,
    double k_speed)
    : model_(&model),
      reference_(&reference),
      k_cte_(k_cte),
      k_heading_(k_heading),
      k_speed_(k_speed) {
    if (reference.states.empty()) {
        throw std::invalid_argument("RolloutController requires a non-empty reference trajectory");
    }
    if (reference.states.front().size() < 4) {
        throw std::invalid_argument("Reference states must contain at least [x, y, yaw, v]");
    }
}

Trajectory RolloutController::generate_initial_trajectory(const Eigen::VectorXd& x0, int horizon) const {
    if (x0.size() < 4) {
        throw std::invalid_argument("Initial state must contain at least [x, y, yaw, v]");
    }
    if (horizon < 0) {
        throw std::invalid_argument("Horizon must be non-negative");
    }

    Trajectory traj;
    traj.states.reserve(static_cast<size_t>(horizon) + 1);
    traj.controls.reserve(static_cast<size_t>(horizon));
    traj.states.push_back(x0);

    double prev_v_des = reference_->states.front()(3);
    for (int t = 0; t < horizon; ++t) {
        const Eigen::VectorXd& x = traj.states.back();
        const int idx = closest_reference_index_(x);
        const double v_des = reference_->states[static_cast<size_t>(idx)](3);
        const double kappa_des = reference_curvature_(idx);

        const double delta_ff = std::atan(model_->wheelbase() * kappa_des);
        const double a_ff = (v_des - prev_v_des) / model_->dt();

        const double cte = reference_->cross_track_error(x);
        const double e_heading = reference_->heading_error(x);
        const double e_v = x(3) - v_des;

        double a_cmd = a_ff - k_speed_ * e_v;
        double delta_cmd = delta_ff - k_cte_ * cte - k_heading_ * e_heading;
        delta_cmd = wrap_angle(delta_cmd);

        Eigen::VectorXd u(2);
        u << a_cmd, delta_cmd;

        if (model_->control_lower_bound_.size() == 2) {
            u = u.cwiseMax(model_->control_lower_bound_);
        }
        if (model_->control_upper_bound_.size() == 2) {
            u = u.cwiseMin(model_->control_upper_bound_);
        }

        traj.controls.push_back(u);
        traj.states.push_back(model_->dynamics(x, u));
        prev_v_des = v_des;
    }

    return traj;
}

int RolloutController::closest_reference_index_(const Eigen::VectorXd& x) const {
    const double px = x(0);
    const double py = x(1);

    int best_idx = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < reference_->states.size(); ++i) {
        const Eigen::VectorXd& r = reference_->states[i];
        const double dx = px - r(0);
        const double dy = py - r(1);
        const double dist2 = dx * dx + dy * dy;
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_idx = static_cast<int>(i);
        }
    }
    return best_idx;
}

double RolloutController::reference_curvature_(int idx) const {
    const int n = static_cast<int>(reference_->states.size());
    if (n < 2) {
        return 0.0;
    }

    const int i0 = std::clamp(idx, 0, n - 2);
    const int i1 = i0 + 1;
    const Eigen::VectorXd& s0 = reference_->states[static_cast<size_t>(i0)];
    const Eigen::VectorXd& s1 = reference_->states[static_cast<size_t>(i1)];

    const double dx = s1(0) - s0(0);
    const double dy = s1(1) - s0(1);
    const double ds = std::sqrt(dx * dx + dy * dy);
    if (ds < 1e-8) {
        return 0.0;
    }

    const double dyaw = wrap_angle(s1(2) - s0(2));
    return dyaw / ds;
}
