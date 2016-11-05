licenses(["restricted"])

cc_library(
    name = "nccl_manager",
    srcs = ["nccl_manager.cc"],
    hdrs = ["nccl_manager.h"],
    deps = [":nccl_lib",],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl_lib",
    srcs = ["libnccl.so"],
    hdrs = ["nccl.h"],
    deps = ["@local_config_cuda//cuda:cuda_headers",],
    visibility = ["//visibility:public"],
)
