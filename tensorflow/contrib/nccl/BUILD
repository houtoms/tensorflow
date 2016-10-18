# Description:
#   Provides NCCL integration to TF

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "if_cuda")

py_library(
    name = "nccl",
    srcs = if_cuda(["python/ops/nccl.py",
            "__init__.py"]),
    data = [":python/ops/tf_nccl.so"],
    srcs_version = "PY2AND3",
    deps = [
            "//tensorflow/python:framework",
    ],
)

tf_custom_op_library(
    name = "python/ops/tf_nccl.so",
    srcs = if_cuda(["core/kernels/tf_nccl.cc"]),
    deps = ["//tensorflow/contrib/nccl/third_party/nccl:nccl_manager"]
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
