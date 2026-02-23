#include "bicycle_model.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

BicycleModel::BicycleModel(double wheelbase, double dt)
    : wheelbase_(wheelbase), dt_(dt) {
    if (wheelbase_ <= 0.0) {
        throw std::invalid_argument("BicycleModel wheelbase must be > 0");
    }
    if (dt_ <= 0.0) {
        throw std::invalid_argument("BicycleModel dt must be > 0");
    }
    control_lower_bound_ = Eigen::VectorXd::Constant(2, -std::numeric_limits<double>::infinity());
    control_upper_bound_ = Eigen::VectorXd::Constant(2, std::numeric_limits<double>::infinity());
}

BicycleModel::BicycleModel(double wheelbase, double dt, const Eigen::VectorXd& control_lower, const Eigen::VectorXd& control_upper)
    : BicycleModel(wheelbase, dt) {
    validate_and_set_limits_(control_lower, control_upper);
}

Eigen::VectorXd BicycleModel::dynamics(const Eigen::VectorXd x, const Eigen::VectorXd u) {
    if (x.size() != 4) {
        throw std::invalid_argument("BicycleModel state must have size 4: [x, y, yaw, v]");
    }
    if (u.size() != 2) {
        throw std::invalid_argument("BicycleModel control must have size 2: [a, delta]");
    }

    const double px = x(0);
    const double py = x(1);
    const double yaw = x(2);
    const double v = x(3);

    const double a = u(0);
    const double delta = u(1);

    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    const double tan_delta = std::tan(delta);

    Eigen::VectorXd next(4);
    next(0) = px + dt_ * v * cos_yaw;
    next(1) = py + dt_ * v * sin_yaw;
    next(2) = yaw + dt_ * v / wheelbase_ * tan_delta;
    next(3) = v + dt_ * a;
    return next;
}

Eigen::MatrixXd BicycleModel::state_jacobian(const Eigen::VectorXd x, const Eigen::VectorXd u) {
    if (x.size() != 4 || u.size() != 2) {
        throw std::invalid_argument("BicycleModel jacobian input sizes must be x:4, u:2");
    }

    const double yaw = x(2);
    const double v = x(3);
    const double delta = u(1);

    const double cos_yaw = std::cos(yaw);
    const double sin_yaw = std::sin(yaw);
    const double tan_delta = std::tan(delta);

    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(4, 4);
    A(0, 2) = -dt_ * v * sin_yaw;
    A(0, 3) = dt_ * cos_yaw;
    A(1, 2) = dt_ * v * cos_yaw;
    A(1, 3) = dt_ * sin_yaw;
    A(2, 3) = dt_ * tan_delta / wheelbase_;
    return A;
}

Eigen::MatrixXd BicycleModel::control_jacobian(const Eigen::VectorXd x, const Eigen::VectorXd u) {
    if (x.size() != 4 || u.size() != 2) {
        throw std::invalid_argument("BicycleModel jacobian input sizes must be x:4, u:2");
    }

    const double v = x(3);
    const double delta = u(1);
    const double sec_delta = 1.0 / std::cos(delta);
    const double sec2_delta = sec_delta * sec_delta;

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(4, 2);
    B(2, 1) = dt_ * v / wheelbase_ * sec2_delta;
    B(3, 0) = dt_;
    return B;
}

void BicycleModel::set_control_limits(const Eigen::VectorXd& control_lower, const Eigen::VectorXd& control_upper) {
    validate_and_set_limits_(control_lower, control_upper);
}

void BicycleModel::set_dt(double dt) {
    if (dt <= 0.0) {
        throw std::invalid_argument("BicycleModel dt must be > 0");
    }
    dt_ = dt;
}

void BicycleModel::set_from_json_string(const std::string& json_text) {
    const std::string section_key = "\"bicycle_model\"";
    const size_t section_pos = json_text.find(section_key);
    std::string text = json_text;
    if (section_pos != std::string::npos) {
        const size_t open_brace = json_text.find('{', section_pos);
        if (open_brace == std::string::npos) {
            throw std::invalid_argument("Invalid JSON: bicycle_model section missing '{'");
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
            throw std::invalid_argument("Invalid JSON: bicycle_model section missing '}'");
        }
        text = json_text.substr(open_brace, close_brace - open_brace + 1);
    }

    {
        const std::regex re_wheelbase("\"wheel_base\"\\s*:\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)");
        std::smatch m;
        if (std::regex_search(text, m, re_wheelbase)) {
            wheelbase_ = std::stod(m[1].str());
        }
    }

    Eigen::VectorXd lower = control_lower_bound_;
    Eigen::VectorXd upper = control_upper_bound_;
    {
        const std::regex re_lower("\"control_lower_bound\"\\s*:\\s*\\[\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*,\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*\\]");
        std::smatch m;
        if (std::regex_search(text, m, re_lower)) {
            lower = Eigen::VectorXd(2);
            lower << std::stod(m[1].str()), std::stod(m[2].str());
        }
    }
    {
        const std::regex re_upper("\"control_upper_bound\"\\s*:\\s*\\[\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*,\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*\\]");
        std::smatch m;
        if (std::regex_search(text, m, re_upper)) {
            upper = Eigen::VectorXd(2);
            upper << std::stod(m[1].str()), std::stod(m[2].str());
        }
    }

    if (wheelbase_ <= 0.0) {
        throw std::invalid_argument("BicycleModel wheelbase must be > 0");
    }
    validate_and_set_limits_(lower, upper);
}

void BicycleModel::set_from_json_file(const std::string& json_file_path) {
    std::ifstream ifs(json_file_path);
    if (!ifs.is_open()) {
        throw std::invalid_argument("Failed to open JSON file: " + json_file_path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    set_from_json_string(oss.str());
}

void BicycleModel::validate_and_set_limits_(const Eigen::VectorXd& control_lower, const Eigen::VectorXd& control_upper) {
    if (control_lower.size() != 2 || control_upper.size() != 2) {
        throw std::invalid_argument("BicycleModel control limits must have size 2: [a, delta]");
    }
    if ((control_lower.array() > control_upper.array()).any()) {
        throw std::invalid_argument("BicycleModel control lower limits must be <= upper limits");
    }
    control_lower_bound_ = control_lower;
    control_upper_bound_ = control_upper;
}
