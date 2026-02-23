#include "bicycle_tracking_cost.h"

#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

BicycleTrackingCost::BicycleTrackingCost(
    const Trajectory& reference,
    double w_cte,
    double w_heading,
    const Eigen::Matrix2d& R,
    double w_cte_terminal,
    double w_heading_terminal,
    double w_speed,
    double w_speed_terminal)
    : reference_(&reference),
      w_cte_(w_cte),
      w_heading_(w_heading),
      w_speed_(w_speed),
      w_cte_terminal_(w_cte_terminal >= 0.0 ? w_cte_terminal : w_cte),
      w_heading_terminal_(w_heading_terminal >= 0.0 ? w_heading_terminal : w_heading),
      w_speed_terminal_(w_speed_terminal >= 0.0 ? w_speed_terminal : w_speed),
      R_(R) {
    if (reference.states.empty() || reference.states.front().size() < 4) {
        throw std::invalid_argument("Reference trajectory must contain at least one state with [x, y, yaw, v]");
    }
    validate_weights_();
}

void BicycleTrackingCost::set_weights_from_json_string(const std::string& json_text) {
    std::string text = json_text;
    const std::string section_name = "\"bicycle_tracking_cost\"";
    const size_t section_pos = text.find(section_name);
    if (section_pos != std::string::npos) {
        const size_t open_brace = text.find('{', section_pos);
        if (open_brace == std::string::npos) {
            throw std::invalid_argument("Invalid JSON: " + section_name + " section missing '{'");
        }
        int depth = 0;
        size_t close_brace = std::string::npos;
        for (size_t i = open_brace; i < text.size(); ++i) {
            if (text[i] == '{') {
                ++depth;
            } else if (text[i] == '}') {
                --depth;
                if (depth == 0) {
                    close_brace = i;
                    break;
                }
            }
        }
        if (close_brace == std::string::npos) {
            throw std::invalid_argument("Invalid JSON: " + section_name + " section missing '}'");
        }
        text = text.substr(open_brace, close_brace - open_brace + 1);
    }

    auto update_scalar_if_present = [&](std::initializer_list<const char*> keys, double& value) {
        for (const char* key : keys) {
            const std::regex re(std::string("\"") + key + "\"\\s*:\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)");
            std::smatch m;
            if (std::regex_search(text, m, re)) {
                value = std::stod(m[1].str());
                return;
            }
        }
    };

    update_scalar_if_present({"w_cte", "cross_track_error_weight"}, w_cte_);
    update_scalar_if_present({"w_heading", "heading_error_weight"}, w_heading_);
    update_scalar_if_present({"w_speed", "speed_error_weight"}, w_speed_);
    update_scalar_if_present({"w_cte_terminal", "cross_track_error_terminal_weight"}, w_cte_terminal_);
    update_scalar_if_present({"w_heading_terminal", "heading_error_terminal_weight"}, w_heading_terminal_);
    update_scalar_if_present({"w_speed_terminal", "speed_error_terminal_weight"}, w_speed_terminal_);

    {
        for (const char* key : {"R", "weight_matrix_controls"}) {
            const std::regex re_r(std::string("\"") + key + "\"\\s*:\\s*\\[\\s*\\[\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*,\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*\\]\\s*,\\s*\\[\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*,\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*\\]\\s*\\]");
            std::smatch m;
            if (std::regex_search(text, m, re_r)) {
                R_(0, 0) = std::stod(m[1].str());
                R_(0, 1) = std::stod(m[2].str());
                R_(1, 0) = std::stod(m[3].str());
                R_(1, 1) = std::stod(m[4].str());
                break;
            }
        }
    }

    validate_weights_();
}

void BicycleTrackingCost::set_weights_from_json_file(const std::string& json_file_path) {
    std::ifstream ifs(json_file_path);
    if (!ifs.is_open()) {
        throw std::invalid_argument("Failed to open JSON file: " + json_file_path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    set_weights_from_json_string(oss.str());
}

void BicycleTrackingCost::validate_weights_() const {
    if (w_cte_ < 0.0 || w_heading_ < 0.0 || w_speed_ < 0.0 ||
        w_cte_terminal_ < 0.0 || w_heading_terminal_ < 0.0 || w_speed_terminal_ < 0.0) {
        throw std::invalid_argument("Cost weights must be non-negative");
    }
    if (!R_.isApprox(R_.transpose(), 1e-12)) {
        throw std::invalid_argument("Control weight matrix R must be symmetric");
    }
}

double BicycleTrackingCost::compute_running_cost(const Eigen::VectorXd x, const Eigen::VectorXd u) {
    validate_state_(x);
    validate_control_(u);

    const double cte = reference_->cross_track_error(x);
    const double he = reference_->heading_error(x);
    const double se = speed_error_(x);
    const double control_cost = u.transpose() * R_ * u;
    return 0.5 * (w_cte_ * cte * cte + w_heading_ * he * he + w_speed_ * se * se + control_cost);
}

double BicycleTrackingCost::compute_terminal_cost(const Eigen::VectorXd x) {
    validate_state_(x);
    const double cte = reference_->cross_track_error(x);
    const double he = reference_->heading_error(x);
    const double se = speed_error_(x);
    return 0.5 * (w_cte_terminal_ * cte * cte + w_heading_terminal_ * he * he + w_speed_terminal_ * se * se);
}

Eigen::VectorXd BicycleTrackingCost::terminal_cost_gradient(const Eigen::VectorXd x) {
    validate_state_(x);
    const double cte = reference_->cross_track_error(x);
    const double he = reference_->heading_error(x);
    const double se = speed_error_(x);
    const Eigen::VectorXd j_cte = reference_->cross_track_error_jacobian(x);
    const Eigen::VectorXd j_he = reference_->heading_error_jacobian(x);
    Eigen::VectorXd grad = w_cte_terminal_ * cte * j_cte + w_heading_terminal_ * he * j_he;
    grad(3) += w_speed_terminal_ * se;
    return grad;
}

Eigen::MatrixXd BicycleTrackingCost::terminal_cost_hessian(const Eigen::VectorXd x) {
    validate_state_(x);
    const Eigen::VectorXd j_cte = reference_->cross_track_error_jacobian(x);
    const Eigen::VectorXd j_he = reference_->heading_error_jacobian(x);
    Eigen::MatrixXd hess = w_cte_terminal_ * (j_cte * j_cte.transpose()) +
                           w_heading_terminal_ * (j_he * j_he.transpose());
    hess(3, 3) += w_speed_terminal_;
    return hess;
}

Eigen::VectorXd BicycleTrackingCost::compute_Lx(const Eigen::VectorXd x, const Eigen::VectorXd u) {
    validate_state_(x);
    validate_control_(u);
    const double cte = reference_->cross_track_error(x);
    const double he = reference_->heading_error(x);
    const double se = speed_error_(x);
    const Eigen::VectorXd j_cte = reference_->cross_track_error_jacobian(x);
    const Eigen::VectorXd j_he = reference_->heading_error_jacobian(x);
    Eigen::VectorXd grad = w_cte_ * cte * j_cte + w_heading_ * he * j_he;
    grad(3) += w_speed_ * se;
    return grad;
}

Eigen::VectorXd BicycleTrackingCost::compute_Lu(const Eigen::VectorXd x, const Eigen::VectorXd u) {
    validate_state_(x);
    validate_control_(u);
    return R_ * u;
}

Eigen::MatrixXd BicycleTrackingCost::compute_Lxx(const Eigen::VectorXd x, const Eigen::VectorXd u) {
    validate_state_(x);
    validate_control_(u);
    const Eigen::VectorXd j_cte = reference_->cross_track_error_jacobian(x);
    const Eigen::VectorXd j_he = reference_->heading_error_jacobian(x);
    Eigen::MatrixXd hess = w_cte_ * (j_cte * j_cte.transpose()) +
                           w_heading_ * (j_he * j_he.transpose());
    hess(3, 3) += w_speed_;
    return hess;
}

Eigen::MatrixXd BicycleTrackingCost::compute_Luu(const Eigen::VectorXd x, const Eigen::VectorXd u) {
    validate_state_(x);
    validate_control_(u);
    return R_;
}

Eigen::MatrixXd BicycleTrackingCost::compute_Lux(const Eigen::VectorXd x, const Eigen::VectorXd u) {
    validate_state_(x);
    validate_control_(u);
    return Eigen::MatrixXd::Zero(u.size(), x.size());
}

void BicycleTrackingCost::validate_state_(const Eigen::VectorXd& x) const {
    if (x.size() < 4) {
        throw std::invalid_argument("BicycleTrackingCost state must have at least [x, y, yaw, v]");
    }
}

void BicycleTrackingCost::validate_control_(const Eigen::VectorXd& u) const {
    if (u.size() != 2) {
        throw std::invalid_argument("BicycleTrackingCost control must have size 2: [a, delta]");
    }
}

double BicycleTrackingCost::speed_error_(const Eigen::VectorXd& x) const {
    double best_dist2 = std::numeric_limits<double>::infinity();
    double ref_speed = reference_->states.front()(3);
    for (const Eigen::VectorXd& rs : reference_->states) {
        if (rs.size() < 4) {
            continue;
        }
        const double dx = x(0) - rs(0);
        const double dy = x(1) - rs(1);
        const double dist2 = dx * dx + dy * dy;
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            ref_speed = rs(3);
        }
    }
    return x(3) - ref_speed;
}
