#pragma once

#include <eigen3/Eigen/Dense>
#include <string>

#include "cost_function.h"
#include "trajectory.h"

// Running cost:
// 0.5 * (w_cte * cte^2 + w_heading * heading_error^2 + w_speed * speed_error^2 + u^T R u)
// Terminal cost:
// 0.5 * (w_cte_terminal * cte^2 + w_heading_terminal * heading_error^2 + w_speed_terminal * speed_error^2)
class BicycleTrackingCost : public CostFunction {
public:
    BicycleTrackingCost(
        const Trajectory& reference,
        double w_cte,
        double w_heading,
        const Eigen::Matrix2d& R,
        double w_cte_terminal = -1.0,
        double w_heading_terminal = -1.0,
        double w_speed = 0.0,
        double w_speed_terminal = -1.0);
    ~BicycleTrackingCost() override = default;

    double compute_running_cost(const Eigen::VectorXd x, const Eigen::VectorXd u) override;
    double compute_terminal_cost(const Eigen::VectorXd x) override;

    Eigen::VectorXd terminal_cost_gradient(const Eigen::VectorXd x) override;
    Eigen::MatrixXd terminal_cost_hessian(const Eigen::VectorXd x) override;

    Eigen::VectorXd compute_Lx(const Eigen::VectorXd x, const Eigen::VectorXd u) override;
    Eigen::VectorXd compute_Lu(const Eigen::VectorXd x, const Eigen::VectorXd u) override;

    Eigen::MatrixXd compute_Lxx(const Eigen::VectorXd x, const Eigen::VectorXd u) override;
    Eigen::MatrixXd compute_Luu(const Eigen::VectorXd x, const Eigen::VectorXd u) override;
    Eigen::MatrixXd compute_Lux(const Eigen::VectorXd x, const Eigen::VectorXd u) override;
    const Trajectory& reference() const { return *reference_; }
    void set_weights_from_json_string(const std::string& json_text);
    void set_weights_from_json_file(const std::string& json_file_path);

private:
    void validate_state_(const Eigen::VectorXd& x) const;
    void validate_control_(const Eigen::VectorXd& u) const;
    double speed_error_(const Eigen::VectorXd& x) const;
    void validate_weights_() const;

    const Trajectory* reference_;
    double w_cte_;
    double w_heading_;
    double w_speed_;
    double w_cte_terminal_;
    double w_heading_terminal_;
    double w_speed_terminal_;
    Eigen::Matrix2d R_;
};
