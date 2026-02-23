#include <iostream>
#include <algorithm>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include "ilqr.h"
#include "bicycle_model.h"
#include "bicycle_tracking_cost.h"
#include "rollout_controller.h"

ILQR::ILQR()
    : horizon_(100),
      step_interval_(0.1),
      tol_(1e-6),
      max_iterations_(100),
      alpha_(1.0),
      armijo_coefficient_(1e-4),
      backtracking_decay_(0.5),
      minimum_step_(1e-6),
      regularization_scale_(1e-6),
      max_projected_newton_iterations_(10),
      active_set_tolerance_(1e-10),
      step_tolerance_(1e-9),
      ptr_model_(nullptr),
      ptr_cost_(nullptr) {
    std::cout << "initialize class ILQR" << std::endl;
    validate_solver_parameters_();
}

ILQR::ILQR(MotionModel& model, CostFunction& cost) 
    : horizon_(100),
      step_interval_(0.1),
      tol_(1e-6),
      max_iterations_(100),
      alpha_(1.0),
      armijo_coefficient_(1e-4),
      backtracking_decay_(0.5),
      minimum_step_(1e-6),
      regularization_scale_(1e-6),
      max_projected_newton_iterations_(10),
      active_set_tolerance_(1e-10),
      step_tolerance_(1e-9),
      ptr_model_(&model),
      ptr_cost_(&cost) {
    validate_solver_parameters_();
    sync_step_interval_to_model_();
}

ILQR::~ILQR(){

}

void ILQR::set_from_json_string(const std::string& json_text) {
    const std::string section_name = "\"ilqr_solver\"";
    const size_t section_pos = json_text.find(section_name);
    std::string text = json_text;
    if (section_pos != std::string::npos) {
        const size_t open_brace = json_text.find('{', section_pos);
        if (open_brace == std::string::npos) {
            throw std::invalid_argument("Invalid JSON: ilqr_solver section missing '{'");
        }
        int depth = 0;
        size_t close_brace = std::string::npos;
        for (size_t i = open_brace; i < json_text.size(); ++i) {
            if (json_text[i] == '{') {
                ++depth;
            } else if (json_text[i] == '}') {
                --depth;
                if (depth == 0) {
                    close_brace = i;
                    break;
                }
            }
        }
        if (close_brace == std::string::npos) {
            throw std::invalid_argument("Invalid JSON: ilqr_solver section missing '}'");
        }
        text = json_text.substr(open_brace, close_brace - open_brace + 1);
    }

    auto update_double_if_present = [&](const char* key, double& value) {
        const std::regex re(std::string("\"") + key + "\"\\s*:\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)");
        std::smatch m;
        if (std::regex_search(text, m, re)) {
            value = std::stod(m[1].str());
        }
    };
    auto update_int_if_present = [&](const char* key, int& value) {
        const std::regex re(std::string("\"") + key + "\"\\s*:\\s*(\\d+)");
        std::smatch m;
        if (std::regex_search(text, m, re)) {
            value = std::stoi(m[1].str());
        }
    };

    update_int_if_present("horizon", horizon_);
    update_double_if_present("step_interval", step_interval_);
    update_int_if_present("max_iterations", max_iterations_);
    update_double_if_present("converge_tolerance", tol_);
    update_double_if_present("learning_rate", alpha_);
    update_double_if_present("armijo_coefficient", armijo_coefficient_);
    update_double_if_present("backtracking_decay", backtracking_decay_);
    update_double_if_present("minimum_step", minimum_step_);
    update_double_if_present("regularization_scale", regularization_scale_);
    update_int_if_present("max_projected_newton_iterations", max_projected_newton_iterations_);
    update_double_if_present("active_set_tolerance", active_set_tolerance_);
    update_double_if_present("step_tolerance", step_tolerance_);

    validate_solver_parameters_();
    sync_step_interval_to_model_();
}

void ILQR::set_from_json_file(const std::string& json_file_path) {
    std::ifstream ifs(json_file_path);
    if (!ifs.is_open()) {
        throw std::invalid_argument("Failed to open JSON file: " + json_file_path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    set_from_json_string(oss.str());
}

ILQR::SolveOutput ILQR::get_solve_output() const {
    SolveOutput out;
    out.initial_rollout_trajectory = initial_rollout_solution_;
    out.final_optimized_trajectory = final_optimized_solution_;

    if (ptr_cost_ != nullptr) {
        auto* tracking_cost = dynamic_cast<BicycleTrackingCost*>(ptr_cost_);
        if (tracking_cost != nullptr) {
            out.reference_trajectory = tracking_cost->reference();
        }
    }
    return out;
}

void ILQR::print_solve_output() const {
    const SolveOutput out = get_solve_output();
    auto print_traj = [](const std::string& name, const Trajectory& traj) {
        std::cout << name << '\n';
        std::cout << "  states (" << traj.states.size() << "):\n";
        for (size_t i = 0; i < traj.states.size(); ++i) {
            std::cout << "    x[" << i << "] = " << traj.states[i].transpose() << '\n';
        }
        std::cout << "  controls (" << traj.controls.size() << "):\n";
        for (size_t i = 0; i < traj.controls.size(); ++i) {
            std::cout << "    u[" << i << "] = " << traj.controls[i].transpose() << '\n';
        }
    };

    print_traj("Reference Trajectory", out.reference_trajectory);
    print_traj("Initial Rollout Trajectory", out.initial_rollout_trajectory);
    print_traj("Final Optimized Trajectory", out.final_optimized_trajectory);
}

void ILQR::save_solve_output_csv(const std::string& path_prefix) const {
    const SolveOutput out = get_solve_output();

    auto write_states_csv = [&](const std::string& file_path, const Trajectory& traj) {
        std::ofstream ofs(file_path);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + file_path);
        }
        ofs << "index";
        const int state_dim = traj.states.empty() ? 0 : static_cast<int>(traj.states.front().size());
        for (int j = 0; j < state_dim; ++j) {
            ofs << ",x" << j;
        }
        ofs << '\n';
        for (size_t i = 0; i < traj.states.size(); ++i) {
            ofs << i;
            for (int j = 0; j < traj.states[i].size(); ++j) {
                ofs << "," << traj.states[i](j);
            }
            ofs << '\n';
        }
    };

    auto write_controls_csv = [&](const std::string& file_path, const Trajectory& traj) {
        std::ofstream ofs(file_path);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + file_path);
        }
        ofs << "index";
        const int control_dim = traj.controls.empty() ? 0 : static_cast<int>(traj.controls.front().size());
        for (int j = 0; j < control_dim; ++j) {
            ofs << ",u" << j;
        }
        ofs << '\n';
        for (size_t i = 0; i < traj.controls.size(); ++i) {
            ofs << i;
            for (int j = 0; j < traj.controls[i].size(); ++j) {
                ofs << "," << traj.controls[i](j);
            }
            ofs << '\n';
        }
    };

    write_states_csv(path_prefix + "_reference_states.csv", out.reference_trajectory);
    write_controls_csv(path_prefix + "_reference_controls.csv", out.reference_trajectory);
    write_states_csv(path_prefix + "_initial_states.csv", out.initial_rollout_trajectory);
    write_controls_csv(path_prefix + "_initial_controls.csv", out.initial_rollout_trajectory);
    write_states_csv(path_prefix + "_final_states.csv", out.final_optimized_trajectory);
    write_controls_csv(path_prefix + "_final_controls.csv", out.final_optimized_trajectory);
}

void ILQR::solve(Eigen::VectorXd x0){
    sync_step_interval_to_model_();
// 1. Initial Rollout (Initial Guess)
    Trajectory traj = initial_rollout(x0);
    initial_rollout_solution_ = traj;
    double old_cost = compute_total_cost(traj);

    for (int iter = 0; iter < max_iterations_; ++iter) {
        // 2. Backward Pass
        auto [k, K] = backward_pass(traj);

        // 3. Forward Pass with Armijo backtracking line search.
        double alpha = alpha_;
        double directional_derivative = 0.0;
        for (const auto& kt : k) {
            directional_derivative -= kt.squaredNorm();
        }

        Trajectory new_traj = traj;
        double new_cost = old_cost;
        bool accepted = false;
        while (alpha >= minimum_step_) {
            new_traj = forward_pass(traj, k, K, alpha);
            new_cost = compute_total_cost(new_traj);
            const double armijo_rhs = old_cost + armijo_coefficient_ * alpha * directional_derivative;
            if (new_cost <= armijo_rhs) {
                accepted = true;
                break;
            }
            alpha *= backtracking_decay_;
        }
        if (!accepted) {
            break;
        }

        // 4. Convergence Check
        if (std::abs(old_cost - new_cost) < tol_) break;
            
        traj = new_traj;
        old_cost = new_cost;
    }
    final_optimized_solution_ = traj;
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> 
ILQR::backward_pass(const Trajectory& traj) {
    std::vector<Eigen::VectorXd> k(horizon_);
    std::vector<Eigen::MatrixXd> K(horizon_);

    // Terminal cost derivatives
    Eigen::VectorXd Vx = terminal_cost_gradient(traj.states.back());
    Eigen::MatrixXd Vxx = terminal_cost_hessian(traj.states.back());

    for (int t = horizon_ - 1; t >= 0; --t) {
        // Get derivatives of Dynamics (f) and Cost (L) at current (x, u)
        Eigen::MatrixXd fx, fu; 
        get_dynamics_derivatives(traj.states[t], traj.controls[t], fx, fu);
            
        Eigen::VectorXd Lx, Lu;
        Eigen::MatrixXd Lxx, Luu, Lux;
        get_cost_derivatives(traj.states[t], traj.controls[t], Lx, Lu, Lxx, Luu, Lux);

        // Q-function expansion
        Eigen::VectorXd Qx = Lx + fx.transpose() * Vx;
        Eigen::VectorXd Qu = Lu + fu.transpose() * Vx;
        Eigen::MatrixXd Qxx = Lxx + fx.transpose() * Vxx * fx;
        Eigen::MatrixXd Quu = Luu + fu.transpose() * Vxx * fu;
        Eigen::MatrixXd Qux = Lux + fu.transpose() * Vxx * fx;

        // Compute gains with Box-DDP projected Newton on:
        // min_d 0.5 d'Quu d + Qu'd,  s.t. lb <= d <= ub.
        Eigen::MatrixXd Quu_reg = Quu;
        Quu_reg.diagonal().array() += regularization_scale_;
        const int m = static_cast<int>(Quu_reg.rows());

        k[t] = Eigen::VectorXd::Zero(m);
        K[t] = Eigen::MatrixXd::Zero(m, Qux.cols());
        if (m > 0) {
            Eigen::VectorXd lb = Eigen::VectorXd::Constant(m, -std::numeric_limits<double>::infinity());
            Eigen::VectorXd ub = Eigen::VectorXd::Constant(m, std::numeric_limits<double>::infinity());
            if (ptr_model_->control_lower_bound_.size() == m) {
                lb = ptr_model_->control_lower_bound_ - traj.controls[t];
            }
            if (ptr_model_->control_upper_bound_.size() == m) {
                ub = ptr_model_->control_upper_bound_ - traj.controls[t];
            }

            Eigen::VectorXd du = Eigen::VectorXd::Zero(m);
            for (int i = 0; i < m; ++i) {
                du(i) = std::clamp(du(i), lb(i), ub(i));
            }

            std::vector<int> free_idx;
            std::vector<int> active_idx;
            free_idx.reserve(m);
            active_idx.reserve(m);

            for (int iter = 0; iter < max_projected_newton_iterations_; ++iter) {
                const Eigen::VectorXd grad = Quu_reg * du + Qu;
                free_idx.clear();
                active_idx.clear();

                for (int i = 0; i < m; ++i) {
                    const bool at_lower = du(i) <= lb(i) + active_set_tolerance_;
                    const bool at_upper = du(i) >= ub(i) - active_set_tolerance_;
                    const bool lower_active = at_lower && grad(i) > 0.0;
                    const bool upper_active = at_upper && grad(i) < 0.0;
                    if (lower_active || upper_active) {
                        active_idx.push_back(i);
                    } else {
                        free_idx.push_back(i);
                    }
                }

                if (free_idx.empty()) {
                    break;
                }

                const int nf = static_cast<int>(free_idx.size());
                Eigen::MatrixXd Hff = Eigen::MatrixXd::Zero(nf, nf);
                Eigen::VectorXd gf = Eigen::VectorXd::Zero(nf);
                for (int r = 0; r < nf; ++r) {
                    const int ir = free_idx[r];
                    gf(r) = grad(ir);
                    for (int c = 0; c < nf; ++c) {
                        Hff(r, c) = Quu_reg(ir, free_idx[c]);
                    }
                }
                Hff.diagonal().array() += regularization_scale_;

                const Eigen::LDLT<Eigen::MatrixXd> hff_ldlt(Hff);
                Eigen::VectorXd step_f = -hff_ldlt.solve(gf);

                double max_step = 0.0;
                for (int i = 0; i < nf; ++i) {
                    const int idx = free_idx[i];
                    const double updated = std::clamp(du(idx) + step_f(i), lb(idx), ub(idx));
                    const double delta = updated - du(idx);
                    du(idx) = updated;
                    max_step = std::max(max_step, std::abs(delta));
                }
                if (max_step < step_tolerance_) {
                    break;
                }
            }
            k[t] = du;

            const Eigen::VectorXd grad_opt = Quu_reg * du + Qu;
            free_idx.clear();
            for (int i = 0; i < m; ++i) {
                const bool at_lower = du(i) <= lb(i) + active_set_tolerance_;
                const bool at_upper = du(i) >= ub(i) - active_set_tolerance_;
                const bool lower_active = at_lower && grad_opt(i) > 0.0;
                const bool upper_active = at_upper && grad_opt(i) < 0.0;
                if (!(lower_active || upper_active)) {
                    free_idx.push_back(i);
                }
            }

            if (!free_idx.empty()) {
                const int nf = static_cast<int>(free_idx.size());
                Eigen::MatrixXd Hff = Eigen::MatrixXd::Zero(nf, nf);
                Eigen::MatrixXd Quxf = Eigen::MatrixXd::Zero(nf, Qux.cols());
                for (int r = 0; r < nf; ++r) {
                    const int ir = free_idx[r];
                    Quxf.row(r) = Qux.row(ir);
                    for (int c = 0; c < nf; ++c) {
                        Hff(r, c) = Quu_reg(ir, free_idx[c]);
                    }
                }
                Hff.diagonal().array() += regularization_scale_;
                const Eigen::LDLT<Eigen::MatrixXd> hff_ldlt(Hff);
                const Eigen::MatrixXd Kf = -hff_ldlt.solve(Quxf);
                for (int r = 0; r < nf; ++r) {
                    K[t].row(free_idx[r]) = Kf.row(r);
                }
            }
        }

        // Update Value Function for next step (t-1)
        Vx = Qx + K[t].transpose() * Quu * k[t] + K[t].transpose() * Qu + Qux.transpose() * k[t];
        Vxx = Qxx + K[t].transpose() * Quu * K[t] + K[t].transpose() * Qux + Qux.transpose() * K[t];

        Vxx = 0.5 * (Vxx + Vxx.transpose());
    }

    return {k, K};
}

Trajectory ILQR::forward_pass(const Trajectory& current, 
                        const std::vector<Eigen::VectorXd>& k,
                        const std::vector<Eigen::MatrixXd>& K,
                        double alpha) {

    Trajectory next;
    next.states.push_back(current.states[0]);
    for (int t = 0; t < horizon_; ++t) {
        Eigen::VectorXd delta_x = next.states[t] - current.states[t];
        Eigen::VectorXd new_u = current.controls[t] + alpha * k[t] + K[t] * delta_x;
        if (ptr_model_->control_lower_bound_.size() == new_u.size()) {
            new_u = new_u.cwiseMax(ptr_model_->control_lower_bound_);
        }
        if (ptr_model_->control_upper_bound_.size() == new_u.size()) {
            new_u = new_u.cwiseMin(ptr_model_->control_upper_bound_);
        }
            
        next.controls.push_back(new_u);
        next.states.push_back(step_dynamics(next.states[t], new_u)); // Physics rollout
    }

    return next;
}

Trajectory ILQR::initial_rollout(const Eigen::VectorXd x0){
    if (ptr_model_ != nullptr && ptr_cost_ != nullptr) {
        auto* bicycle_model = dynamic_cast<BicycleModel*>(ptr_model_);
        auto* tracking_cost = dynamic_cast<BicycleTrackingCost*>(ptr_cost_);
        if (bicycle_model != nullptr && tracking_cost != nullptr) {
            RolloutController controller(*bicycle_model, tracking_cost->reference());
            return controller.generate_initial_trajectory(x0, horizon_);
        }
    }

    // Generic fallback: zero-control rollout with control dimension inferred from model limits.
    Trajectory traj;
    traj.states.reserve(static_cast<size_t>(horizon_) + 1);
    traj.controls.reserve(static_cast<size_t>(horizon_));
    Eigen::VectorXd x = x0;
    traj.states.push_back(x);
    const int control_dim = (ptr_model_ != nullptr) ? static_cast<int>(ptr_model_->control_lower_bound_.size()) : 0;
    for (int t = 0; t < horizon_; ++t) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(control_dim);
        x = step_dynamics(x, u);
        traj.states.push_back(x);
        traj.controls.push_back(u);
    }
    return traj;
}

double ILQR::compute_total_cost(const Trajectory& traj){
    double cost = 0.0;

    for (int t = 0; t < horizon_; ++t) {
        cost += ptr_cost_->compute_running_cost(traj.states[t], traj.controls[t]);
    }

    cost += ptr_cost_->compute_terminal_cost(traj.states[horizon_]);

    return cost;

}

Eigen::VectorXd ILQR::terminal_cost_gradient(const Eigen::VectorXd x){

    Eigen::VectorXd cost_gradient = ptr_cost_->terminal_cost_gradient(x);
    return cost_gradient;
}

Eigen::MatrixXd ILQR::terminal_cost_hessian(const Eigen::VectorXd x){
    
    Eigen::MatrixXd cost_hessian  = ptr_cost_->terminal_cost_hessian(x);
    return cost_hessian;
}

void ILQR::get_dynamics_derivatives(const Eigen::VectorXd x, 
                                const Eigen::VectorXd u, 
                                Eigen::MatrixXd& fx, 
                                Eigen::MatrixXd& fu){

                                    fx = ptr_model_->state_jacobian(x, u);
                                    fu = ptr_model_->control_jacobian(x, u);
                                }

void ILQR::get_cost_derivatives(const Eigen::VectorXd x, 
                        const Eigen::VectorXd u, 
                        Eigen::VectorXd& Lx, 
                        Eigen::VectorXd& Lu, 
                        Eigen::MatrixXd& Lxx, 
                        Eigen::MatrixXd& Luu, 
                        Eigen::MatrixXd& Lux){

                            Lx = ptr_cost_->compute_Lx(x, u);
                            Lu = ptr_cost_->compute_Lu(x, u);
                            Lxx = ptr_cost_->compute_Lxx(x,u);
                            Luu = ptr_cost_->compute_Luu(x, u);
                            Lux = ptr_cost_->compute_Lux(x, u);
                        }

Eigen::VectorXd ILQR::step_dynamics(const Eigen::VectorXd x, const Eigen::VectorXd u){

    Eigen::VectorXd next_state = ptr_model_->dynamics(x, u);
    return next_state;
}

void ILQR::validate_solver_parameters_() const {
    if (horizon_ <= 0) {
        throw std::invalid_argument("ILQR horizon must be > 0");
    }
    if (step_interval_ <= 0.0) {
        throw std::invalid_argument("ILQR step_interval must be > 0");
    }
    if (max_iterations_ <= 0) {
        throw std::invalid_argument("ILQR max_iterations must be > 0");
    }
    if (tol_ <= 0.0) {
        throw std::invalid_argument("ILQR converge_tolerance must be > 0");
    }
    if (alpha_ <= 0.0) {
        throw std::invalid_argument("ILQR learning_rate must be > 0");
    }
    if (armijo_coefficient_ <= 0.0 || armijo_coefficient_ >= 1.0) {
        throw std::invalid_argument("ILQR armijo_coefficient must be in (0, 1)");
    }
    if (backtracking_decay_ <= 0.0 || backtracking_decay_ >= 1.0) {
        throw std::invalid_argument("ILQR backtracking_decay must be in (0, 1)");
    }
    if (minimum_step_ <= 0.0) {
        throw std::invalid_argument("ILQR minimum_step must be > 0");
    }
    if (regularization_scale_ < 0.0) {
        throw std::invalid_argument("ILQR regularization_scale must be >= 0");
    }
    if (max_projected_newton_iterations_ <= 0) {
        throw std::invalid_argument("ILQR max_projected_newton_iterations must be > 0");
    }
    if (active_set_tolerance_ <= 0.0) {
        throw std::invalid_argument("ILQR active_set_tolerance must be > 0");
    }
    if (step_tolerance_ <= 0.0) {
        throw std::invalid_argument("ILQR step_tolerance must be > 0");
    }
}

void ILQR::sync_step_interval_to_model_() {
    if (ptr_model_ == nullptr) {
        return;
    }
    auto* bicycle_model = dynamic_cast<BicycleModel*>(ptr_model_);
    if (bicycle_model != nullptr) {
        bicycle_model->set_dt(step_interval_);
    }
}


                        
