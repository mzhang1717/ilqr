#include "trajectory.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

void Trajectory::approximate_states_with_cubic_spline() {
    if (states.size() < 2) {
        spline_ready_ = false;
        throw std::invalid_argument("Trajectory requires at least 2 states to fit a spline");
    }

    std::vector<double> px;
    std::vector<double> py;
    px.reserve(states.size());
    py.reserve(states.size());

    for (const Eigen::VectorXd& s : states) {
        if (s.size() < 2) {
            spline_ready_ = false;
            throw std::invalid_argument("Each trajectory state must contain at least [x, y]");
        }
        px.push_back(s(0));
        py.push_back(s(1));
    }

    std::vector<double> keep_x;
    std::vector<double> keep_y;
    keep_x.reserve(px.size());
    keep_y.reserve(py.size());
    keep_x.push_back(px.front());
    keep_y.push_back(py.front());
    for (size_t i = 1; i < px.size(); ++i) {
        const double dx = px[i] - keep_x.back();
        const double dy = py[i] - keep_y.back();
        if (dx * dx + dy * dy > 1e-12) {
            keep_x.push_back(px[i]);
            keep_y.push_back(py[i]);
        }
    }

    if (keep_x.size() < 2) {
        spline_ready_ = false;
        throw std::invalid_argument("Spline fitting requires at least 2 distinct [x, y] points");
    }

    s_.assign(keep_x.size(), 0.0);
    for (size_t i = 1; i < keep_x.size(); ++i) {
        const double dx = keep_x[i] - keep_x[i - 1];
        const double dy = keep_y[i] - keep_y[i - 1];
        s_[i] = s_[i - 1] + std::sqrt(dx * dx + dy * dy);
    }

    h_.assign(s_.size() - 1, 0.0);
    for (size_t i = 0; i + 1 < s_.size(); ++i) {
        h_[i] = s_[i + 1] - s_[i];
    }

    sx_ = fit_natural_cubic_(s_, keep_x);
    sy_ = fit_natural_cubic_(s_, keep_y);
    spline_ready_ = true;
}

double Trajectory::cross_track_error(const Eigen::VectorXd& state) const {
    if (!spline_ready_) {
        throw std::logic_error("Spline is not available; call approximate_states_with_cubic_spline first");
    }
    if (state.size() < 2) {
        throw std::invalid_argument("Query state must contain at least [x, y]");
    }

    const double px = state(0);
    const double py = state(1);

    const ProjectionResult proj = project_to_spline_(px, py);
    const double ex = px - proj.x;
    const double ey = py - proj.y;
    const double unsigned_err = std::sqrt(ex * ex + ey * ey);
    const double cross = proj.dx * ey - proj.dy * ex;
    return (cross >= 0.0) ? unsigned_err : -unsigned_err;
}

double Trajectory::heading_error(const Eigen::VectorXd& state) const {
    if (!spline_ready_) {
        throw std::logic_error("Spline is not available; call approximate_states_with_cubic_spline first");
    }
    if (state.size() < 3) {
        throw std::invalid_argument("Query state must contain at least [x, y, yaw]");
    }

    const ProjectionResult proj = project_to_spline_(state(0), state(1));
    const double spline_yaw = std::atan2(proj.dy, proj.dx);
    return normalize_angle_(state(2) - spline_yaw);
}

Eigen::VectorXd Trajectory::cross_track_error_jacobian(const Eigen::VectorXd& state) const {
    if (!spline_ready_) {
        throw std::logic_error("Spline is not available; call approximate_states_with_cubic_spline first");
    }
    if (state.size() < 2) {
        throw std::invalid_argument("Query state must contain at least [x, y]");
    }

    const ProjectionResult proj = project_to_spline_(state(0), state(1));
    const double tx = proj.dx;
    const double ty = proj.dy;
    const double norm_t = std::sqrt(tx * tx + ty * ty);

    Eigen::VectorXd jac = Eigen::VectorXd::Zero(state.size());
    if (norm_t <= 1e-12) {
        return jac;
    }

    jac(0) = -ty / norm_t;
    jac(1) = tx / norm_t;
    return jac;
}

Eigen::VectorXd Trajectory::heading_error_jacobian(const Eigen::VectorXd& state) const {
    if (!spline_ready_) {
        throw std::logic_error("Spline is not available; call approximate_states_with_cubic_spline first");
    }
    if (state.size() < 3) {
        throw std::invalid_argument("Query state must contain at least [x, y, yaw]");
    }

    Eigen::VectorXd jac = Eigen::VectorXd::Zero(state.size());
    jac(2) = 1.0;
    return jac;
}

double Trajectory::normalize_angle_(double angle) {
    constexpr double kPi = 3.14159265358979323846;
    while (angle > kPi) {
        angle -= 2.0 * kPi;
    }
    while (angle < -kPi) {
        angle += 2.0 * kPi;
    }
    return angle;
}

Trajectory::Cubic1D Trajectory::fit_natural_cubic_(const std::vector<double>& t, const std::vector<double>& y) {
    const size_t n = y.size();
    Cubic1D out;
    out.a.assign(n - 1, 0.0);
    out.b.assign(n - 1, 0.0);
    out.c.assign(n - 1, 0.0);
    out.d.assign(n - 1, 0.0);

    std::vector<double> h(n - 1, 0.0);
    for (size_t i = 0; i + 1 < n; ++i) {
        h[i] = t[i + 1] - t[i];
    }

    std::vector<double> alpha(n, 0.0);
    for (size_t i = 1; i + 1 < n; ++i) {
        alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1]);
    }

    std::vector<double> l(n, 0.0);
    std::vector<double> mu(n, 0.0);
    std::vector<double> z(n, 0.0);
    std::vector<double> cnode(n, 0.0);

    l[0] = 1.0;
    for (size_t i = 1; i + 1 < n; ++i) {
        l[i] = 2.0 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[n - 1] = 1.0;

    for (int j = static_cast<int>(n) - 2; j >= 0; --j) {
        cnode[j] = z[j] - mu[j] * cnode[j + 1];
        out.a[j] = y[j];
        out.b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (cnode[j + 1] + 2.0 * cnode[j]) / 3.0;
        out.c[j] = cnode[j];
        out.d[j] = (cnode[j + 1] - cnode[j]) / (3.0 * h[j]);
    }

    return out;
}

int Trajectory::segment_index_(double s) const {
    if (s <= 0.0) {
        return 0;
    }
    const double s_max = s_.back();
    if (s >= s_max) {
        return static_cast<int>(s_.size()) - 2;
    }
    const auto it = std::upper_bound(s_.begin(), s_.end(), s);
    int idx = static_cast<int>(it - s_.begin()) - 1;
    if (idx < 0) {
        idx = 0;
    }
    const int max_idx = static_cast<int>(s_.size()) - 2;
    if (idx > max_idx) {
        idx = max_idx;
    }
    return idx;
}

void Trajectory::eval_spline_(double s, double& x, double& y, double& dx, double& dy, double& ddx, double& ddy) const {
    const int i = segment_index_(s);
    const double t = std::clamp(s - s_[i], 0.0, h_[i]);

    const double tx = t;
    const double tx2 = tx * tx;

    x = sx_.a[i] + sx_.b[i] * tx + sx_.c[i] * tx2 + sx_.d[i] * tx2 * tx;
    y = sy_.a[i] + sy_.b[i] * tx + sy_.c[i] * tx2 + sy_.d[i] * tx2 * tx;

    dx = sx_.b[i] + 2.0 * sx_.c[i] * tx + 3.0 * sx_.d[i] * tx2;
    dy = sy_.b[i] + 2.0 * sy_.c[i] * tx + 3.0 * sy_.d[i] * tx2;

    ddx = 2.0 * sx_.c[i] + 6.0 * sx_.d[i] * tx;
    ddy = 2.0 * sy_.c[i] + 6.0 * sy_.d[i] * tx;
}

Trajectory::ProjectionResult Trajectory::project_to_spline_(double px, double py) const {
    double best_s = 0.0;
    double best_dist2 = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i + 1 < s_.size(); ++i) {
        const double x0 = sx_.a[i];
        const double y0 = sy_.a[i];
        const double x1 = sx_.a[i] + sx_.b[i] * h_[i] + sx_.c[i] * h_[i] * h_[i] + sx_.d[i] * h_[i] * h_[i] * h_[i];
        const double y1 = sy_.a[i] + sy_.b[i] * h_[i] + sy_.c[i] * h_[i] * h_[i] + sy_.d[i] * h_[i] * h_[i] * h_[i];
        const double vx = x1 - x0;
        const double vy = y1 - y0;
        const double denom = vx * vx + vy * vy;
        const double u = (denom > 1e-12) ? std::clamp(((px - x0) * vx + (py - y0) * vy) / denom, 0.0, 1.0) : 0.0;
        const double s_guess = s_[i] + u * h_[i];

        double gx, gy, gdx, gdy, gddx, gddy;
        eval_spline_(s_guess, gx, gy, gdx, gdy, gddx, gddy);
        const double ex = px - gx;
        const double ey = py - gy;
        const double dist2 = ex * ex + ey * ey;
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_s = s_guess;
        }
    }

    const double s_max = s_.back();
    for (int it = 0; it < 5; ++it) {
        double gx, gy, gdx, gdy, gddx, gddy;
        eval_spline_(best_s, gx, gy, gdx, gdy, gddx, gddy);
        const double rx = gx - px;
        const double ry = gy - py;
        const double grad = rx * gdx + ry * gdy;
        const double hess = gdx * gdx + gdy * gdy + rx * gddx + ry * gddy;
        if (std::abs(hess) < 1e-12) {
            break;
        }
        const double next_s = std::clamp(best_s - grad / hess, 0.0, s_max);
        if (std::abs(next_s - best_s) < 1e-7) {
            best_s = next_s;
            break;
        }
        best_s = next_s;
    }

    ProjectionResult out{};
    double ddx, ddy;
    out.s = best_s;
    eval_spline_(best_s, out.x, out.y, out.dx, out.dy, ddx, ddy);
    return out;
}
