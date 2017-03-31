# NVIDIA nccl
# A package of optimized primitives for collective multi-GPU communication.

licenses(["notice"])  # BSD

exports_files(["LICENSE.txt"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts", "if_cuda")

cc_library(
    name = "nccl",
    hdrs = if_cuda(["src/nccl.h"]),
    copts = [
        "-O3",
    ] + cuda_default_copts(),
    linkopts = select({
        "//conditions:default": [
            "-lrt",
            "-lnccl",
        ],
    }),
    visibility = ["//visibility:public"],
)
