#pragma once

#include <eigen3/Eigen/Dense>
#include "trajectory.h"

class MotionModel {
public:    
    MotionModel(){}
    virtual ~MotionModel(){}

    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd x, const Eigen::VectorXd u) = 0;
    virtual Eigen::MatrixXd state_jacobian(const Eigen::VectorXd x, const Eigen::VectorXd u) = 0;
    virtual Eigen::MatrixXd control_jacobian(const Eigen::VectorXd x, const Eigen::VectorXd u) = 0;

    Eigen::VectorXd control_lower_bound_;
    Eigen::VectorXd control_upper_bound_;

};
