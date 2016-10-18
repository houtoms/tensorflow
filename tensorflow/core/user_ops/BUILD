load("//tensorflow:tensorflow.bzl",
     "tf_custom_op_library")


tf_custom_op_library(
    name = "tf_nccl.so",
    srcs = ["tf_nccl.cc"],
    deps = ["//third_party/nccl:nccl_manager"]
)
