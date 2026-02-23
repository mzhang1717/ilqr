cc_library(
    name = "trajectory",
    srcs = ["trajectory.cpp"],
    hdrs = ["trajectory.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "motion_model",
    hdrs = ["motion_model.h"],
    deps = [":trajectory"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "bicycle_model",
    srcs = ["bicycle_model.cpp"],
    hdrs = ["bicycle_model.h"],
    deps = [":motion_model"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cost_function",
    hdrs = ["cost_function.h"],
    deps = [":motion_model"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "bicycle_tracking_cost",
    srcs = ["bicycle_tracking_cost.cpp"],
    hdrs = ["bicycle_tracking_cost.h"],
    deps = [
        ":cost_function",
        ":trajectory",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rollout_controller",
    srcs = ["rollout_controller.cpp"],
    hdrs = ["rollout_controller.h"],
    deps = [
        ":bicycle_model",
        ":trajectory",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ilqr",
    srcs = ["ilqr.cpp"],
    hdrs = ["ilqr.h"],
    deps = [
        ":bicycle_model",
        ":bicycle_tracking_cost",
        ":cost_function",
        ":motion_model",
        ":rollout_controller",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [":ilqr"],
)

cc_test(
    name = "trajectory_test",
    srcs = ["trajectory_test.cpp"],
    deps = [":trajectory"],
)

cc_test(
    name = "bicycle_model_test",
    srcs = ["bicycle_model_test.cpp"],
    deps = [":bicycle_model"],
)

cc_test(
    name = "bicycle_tracking_cost_test",
    srcs = ["bicycle_tracking_cost_test.cpp"],
    deps = [":bicycle_tracking_cost"],
)

cc_test(
    name = "ilqr_test",
    srcs = ["ilqr_test.cpp"],
    data = ["ilqr_config.json"],
    deps = [
        ":bicycle_model",
        ":bicycle_tracking_cost",
        ":ilqr",
    ],
)
