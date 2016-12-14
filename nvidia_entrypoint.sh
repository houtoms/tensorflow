#!/bin/bash
set -e
cat <<EOF
                                                                                                                                                
================
== TensorFlow ==
================

NVIDIA Release ${NVIDIA_TENSORFLOW_VERSION} (build ${NVIDIA_BUILD_ID})

Container image Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
Copyright 2015 The TensorFlow Authors.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying project or file.

EOF

if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  exec "$@"
fi
