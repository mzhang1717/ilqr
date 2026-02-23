#include "ilqr.h"
#include "bicycle_model.h"
#include "bicycle_tracking_cost.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

class MockMotionModel : public MotionModel {
public:
    int dynamics_calls = 0;
    int state_jacobian_calls = 0;
    int control_jacobian_calls = 0;

    Eigen::VectorXd dynamics(const Eigen::VectorXd x, const Eigen::VectorXd u) override {
        ++dynamics_calls;
        (void)u;
        return Eigen::VectorXd::Zero(x.size());
    }

    Eigen::MatrixXd state_jacobian(const Eigen::VectorXd x, const Eigen::VectorXd u) override {
        ++state_jacobian_calls;
        (void)u;
        return Eigen::MatrixXd::Zero(x.size(), x.size());
    }

    Eigen::MatrixXd control_jacobian(const Eigen::VectorXd x, const Eigen::VectorXd u) override {
        ++control_jacobian_calls;
        return Eigen::MatrixXd::Zero(x.size(), u.size());
    }
};

class MockCostFunction : public CostFunction {
public:
    int running_cost_calls = 0;
    int terminal_cost_calls = 0;
    int terminal_gradient_calls = 0;
    int terminal_hessian_calls = 0;
    int lx_calls = 0;
    int lu_calls = 0;
    int lxx_calls = 0;
    int luu_calls = 0;
    int lux_calls = 0;

    double compute_running_cost(const Eigen::VectorXd x, const Eigen::VectorXd u) override {
        ++running_cost_calls;
        (void)x;
        (void)u;
        return 0.0;
    }

    double compute_terminal_cost(const Eigen::VectorXd x) override {
        ++terminal_cost_calls;
        (void)x;
        return 0.0;
    }

    Eigen::VectorXd terminal_cost_gradient(const Eigen::VectorXd x) override {
        ++terminal_gradient_calls;
        return Eigen::VectorXd::Zero(x.size());
    }

    Eigen::MatrixXd terminal_cost_hessian(const Eigen::VectorXd x) override {
        ++terminal_hessian_calls;
        return Eigen::MatrixXd::Zero(x.size(), x.size());
    }

    Eigen::VectorXd compute_Lx(const Eigen::VectorXd x, const Eigen::VectorXd u) override {
        ++lx_calls;
        (void)u;
        return Eigen::VectorXd::Zero(x.size());
    }

    Eigen::VectorXd compute_Lu(const Eigen::VectorXd x, const Eigen::VectorXd u) override {
        ++lu_calls;
        (void)x;
        return Eigen::VectorXd::Zero(u.size());
    }

    Eigen::MatrixXd compute_Lxx(const Eigen::VectorXd x, const Eigen::VectorXd u) override {
        ++lxx_calls;
        (void)u;
        return Eigen::MatrixXd::Zero(x.size(), x.size());
    }

    Eigen::MatrixXd compute_Luu(const Eigen::VectorXd x, const Eigen::VectorXd u) override {
        ++luu_calls;
        (void)x;
        return Eigen::MatrixXd::Zero(u.size(), u.size());
    }

    Eigen::MatrixXd compute_Lux(const Eigen::VectorXd x, const Eigen::VectorXd u) override {
        ++lux_calls;
        return Eigen::MatrixXd::Zero(u.size(), x.size());
    }
};

int run_mock_stack_test() {
    constexpr int kHorizon = 100;
    MockMotionModel model;
    MockCostFunction cost;
    ILQR solver(model, cost);

    Eigen::VectorXd x0(0);
    solver.solve(x0);
    solver.print_solve_output();
    solver.save_solve_output_csv("/tmp/ilqr_mock_test");

    const int expected_dynamics_calls = 2 * kHorizon;  // initial rollout + one accepted forward pass
    const int expected_derivative_calls = kHorizon;    // one backward pass
    const int expected_running_cost_calls = 2 * kHorizon;  // old_cost + new_cost

    if (model.dynamics_calls != expected_dynamics_calls) {
        std::cerr << "Unexpected dynamics call count: " << model.dynamics_calls
                  << " (want " << expected_dynamics_calls << ")\n";
        return 1;
    }
    if (model.state_jacobian_calls != expected_derivative_calls) {
        std::cerr << "Unexpected state_jacobian call count: " << model.state_jacobian_calls
                  << " (want " << expected_derivative_calls << ")\n";
        return 1;
    }
    if (model.control_jacobian_calls != expected_derivative_calls) {
        std::cerr << "Unexpected control_jacobian call count: " << model.control_jacobian_calls
                  << " (want " << expected_derivative_calls << ")\n";
        return 1;
    }

    if (cost.running_cost_calls != expected_running_cost_calls) {
        std::cerr << "Unexpected running_cost call count: " << cost.running_cost_calls
                  << " (want " << expected_running_cost_calls << ")\n";
        return 1;
    }
    if (cost.terminal_cost_calls != 2) {
        std::cerr << "Unexpected terminal_cost call count: " << cost.terminal_cost_calls << " (want 2)\n";
        return 1;
    }
    if (cost.terminal_gradient_calls != 1 || cost.terminal_hessian_calls != 1) {
        std::cerr << "Unexpected terminal derivative call counts: gradient=" << cost.terminal_gradient_calls
                  << ", hessian=" << cost.terminal_hessian_calls << " (want 1,1)\n";
        return 1;
    }
    if (cost.lx_calls != expected_derivative_calls || cost.lu_calls != expected_derivative_calls ||
        cost.lxx_calls != expected_derivative_calls || cost.luu_calls != expected_derivative_calls ||
        cost.lux_calls != expected_derivative_calls) {
        std::cerr << "Unexpected running derivative call counts: "
                  << "Lx=" << cost.lx_calls << ", Lu=" << cost.lu_calls << ", Lxx=" << cost.lxx_calls
                  << ", Luu=" << cost.luu_calls << ", Lux=" << cost.lux_calls
                  << " (want all " << expected_derivative_calls << ")\n";
        return 1;
    }

    return 0;
}

int run_bicycle_stack_test() {
    std::string config_path = "ilqr_config.json";
    if (const char* test_srcdir = std::getenv("TEST_SRCDIR")) {
        const char* test_workspace = std::getenv("TEST_WORKSPACE");
        const std::string workspace = (test_workspace != nullptr) ? test_workspace : "_main";
        config_path = std::string(test_srcdir) + "/" + workspace + "/ilqr_config.json";
    }

    Trajectory reference;
    for (int i = 0; i <= 20; ++i) {
        Eigen::VectorXd s(4);
        s << static_cast<double>(i) * 0.5, 0.0, 0.0, 1.0;
        reference.states.push_back(s);
    }
    reference.approximate_states_with_cubic_spline();

    BicycleModel model(0.5, 0.1);
    model.set_from_json_file(config_path);
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
    BicycleTrackingCost cost(reference, 10.0, 2.0, R, 20.0, 4.0, 10.0, 20.0);
    cost.set_weights_from_json_file(config_path);
    ILQR solver(model, cost);
    solver.set_from_json_file(config_path);

    Eigen::VectorXd x0(4);
    x0 << 0.0, 0.3, -0.05, 0.0;

    bool threw = false;
    try {
        solver.solve(x0);
        solver.print_solve_output();
        solver.save_solve_output_csv("/tmp/ilqr_bicycle_test");
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    if (threw) {
        std::cerr << "ILQR::solve unexpectedly threw for BicycleModel + BicycleTrackingCost stack.\n";
        return 1;
    }

    return 0;
}

}  // namespace

int main() {
    const int mock_status = run_mock_stack_test();
    if (mock_status != 0) {
        return mock_status;
    }
    return run_bicycle_stack_test();
}
