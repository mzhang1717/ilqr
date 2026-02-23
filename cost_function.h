#pragma once

#include <eigen3/Eigen/Dense>
#include "motion_model.h"

class CostFunction {
public:
    CostFunction() {}
    virtual ~CostFunction() {}

    virtual double compute_running_cost(const Eigen::VectorXd x, const Eigen::VectorXd u) = 0;
    virtual double compute_terminal_cost(const Eigen::VectorXd x) = 0;

    virtual Eigen::VectorXd terminal_cost_gradient(const Eigen::VectorXd x)  = 0;
    virtual Eigen::MatrixXd terminal_cost_hessian(const Eigen::VectorXd x) = 0;

    virtual Eigen::VectorXd compute_Lx(const Eigen::VectorXd x, const Eigen::VectorXd u) = 0;
    virtual Eigen::VectorXd compute_Lu(const Eigen::VectorXd x, const Eigen::VectorXd u) = 0;
    
    virtual Eigen::MatrixXd compute_Lxx(const Eigen::VectorXd x, const Eigen::VectorXd u) = 0;
    virtual Eigen::MatrixXd compute_Luu(const Eigen::VectorXd x, const Eigen::VectorXd u) = 0;
    virtual Eigen::MatrixXd compute_Lux(const Eigen::VectorXd x, const Eigen::VectorXd u) = 0;

    MotionModel* prt_model_;
};