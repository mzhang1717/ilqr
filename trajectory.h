#pragma once

#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Dense>

class Trajectory {
public:
    std::vector<Eigen::VectorXd> states;
    std::vector<Eigen::VectorXd> controls;

    void approximate_states_with_cubic_spline();
    double cross_track_error(const Eigen::VectorXd& state) const;
    double heading_error(const Eigen::VectorXd& state) const;
    Eigen::VectorXd cross_track_error_jacobian(const Eigen::VectorXd& state) const;
    Eigen::VectorXd heading_error_jacobian(const Eigen::VectorXd& state) const;

private:
    struct Cubic1D {
        std::vector<double> a;
        std::vector<double> b;
        std::vector<double> c;
        std::vector<double> d;
    };

    struct ProjectionResult {
        double s;
        double x;
        double y;
        double dx;
        double dy;
    };

    static double normalize_angle_(double angle);
    static Cubic1D fit_natural_cubic_(const std::vector<double>& t, const std::vector<double>& y);

    int segment_index_(double s) const;
    void eval_spline_(double s, double& x, double& y, double& dx, double& dy, double& ddx, double& ddy) const;
    ProjectionResult project_to_spline_(double px, double py) const;

    bool spline_ready_ = false;
    std::vector<double> s_;
    std::vector<double> h_;
    Cubic1D sx_;
    Cubic1D sy_;
};
