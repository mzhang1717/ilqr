#pragma once

#include <eigen3/Eigen/Dense>
#include <string>

#include "motion_model.h"

// State: [x, y, yaw, v], Control: [a, delta]
class BicycleModel : public MotionModel {
public:
    BicycleModel(double wheelbase, double dt);
    BicycleModel(double wheelbase, double dt, const Eigen::VectorXd& control_lower, const Eigen::VectorXd& control_upper);
    ~BicycleModel() override = default;

    Eigen::VectorXd dynamics(const Eigen::VectorXd x, const Eigen::VectorXd u) override;
    Eigen::MatrixXd state_jacobian(const Eigen::VectorXd x, const Eigen::VectorXd u) override;
    Eigen::MatrixXd control_jacobian(const Eigen::VectorXd x, const Eigen::VectorXd u) override;
    void set_control_limits(const Eigen::VectorXd& control_lower, const Eigen::VectorXd& control_upper);
    void set_dt(double dt);
    void set_from_json_string(const std::string& json_text);
    void set_from_json_file(const std::string& json_file_path);
    double wheelbase() const { return wheelbase_; }
    double dt() const { return dt_; }

private:
    void validate_and_set_limits_(const Eigen::VectorXd& control_lower, const Eigen::VectorXd& control_upper);

    double wheelbase_;
    double dt_;
};
