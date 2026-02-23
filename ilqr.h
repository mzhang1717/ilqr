#pragma once

#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include "motion_model.h"
#include "cost_function.h"

class ILQR {
public:
    struct SolveOutput {
        Trajectory reference_trajectory;
        Trajectory initial_rollout_trajectory;
        Trajectory final_optimized_trajectory;
    };

    ILQR();
    ILQR(MotionModel& model, CostFunction& cost);
    ~ILQR();

    void solve (Eigen::VectorXd x0);
    SolveOutput get_solve_output() const;
    void print_solve_output() const;
    void save_solve_output_csv(const std::string& path_prefix) const;
    void set_from_json_string(const std::string& json_text);
    void set_from_json_file(const std::string& json_file_path);

private:
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backward_pass(const Trajectory& traj);

    Trajectory forward_pass(const Trajectory& current, 
                            const std::vector<Eigen::VectorXd>& k,
                            const std::vector<Eigen::MatrixXd>& K,
                            double alpha);
    
    Trajectory initial_rollout(const Eigen::VectorXd x0);
    double compute_total_cost(const Trajectory& traj);

    Eigen::VectorXd terminal_cost_gradient(const Eigen::VectorXd x);
    Eigen::MatrixXd terminal_cost_hessian(const Eigen::VectorXd x);

    void get_dynamics_derivatives(const Eigen::VectorXd x, 
                                    const Eigen::VectorXd u, 
                                    Eigen::MatrixXd& fx, 
                                    Eigen::MatrixXd& fu);

    void get_cost_derivatives(const Eigen::VectorXd x, 
                            const Eigen::VectorXd u, 
                            Eigen::VectorXd& Lx, 
                            Eigen::VectorXd& Lu, 
                            Eigen::MatrixXd& Lxx, 
                            Eigen::MatrixXd& Luu, 
                            Eigen::MatrixXd& Lux);

    Eigen::VectorXd step_dynamics(const Eigen::VectorXd x, const Eigen::VectorXd u);
    void validate_solver_parameters_() const;
    void sync_step_interval_to_model_();
    
    int horizon_;
    double step_interval_;
    double tol_;
    int max_iterations_;
    double alpha_ ; // Learning rate (step size)
    double armijo_coefficient_;
    double backtracking_decay_;
    double minimum_step_;
    double regularization_scale_;
    int max_projected_newton_iterations_;
    double active_set_tolerance_;
    double step_tolerance_;
    Trajectory initial_rollout_solution_;
    Trajectory final_optimized_solution_;

    MotionModel* ptr_model_;
    CostFunction* ptr_cost_;
};
