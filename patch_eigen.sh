#!/bin/bash

cd /root/.cache/bazel/_bazel_root/*/external/eigen_archive
if [[ $? -ne 0 ]]; then
    echo "Could not find eigen in cache"
    exit 1
fi

patch -p1 --dry-run -N --silent < /opt/tensorflow/eigen.redux.patch
if [[ $? -ne 0 ]]; then
    echo "Skipping eigen redux patch"
    exit 0
fi

patch -p1 -N < /opt/tensorflow/eigen.redux.patch
exit $?
