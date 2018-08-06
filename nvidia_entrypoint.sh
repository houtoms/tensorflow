#!/bin/bash
set -e
cat <<EOF
                                                                                                                                                
================
== TensorFlow ==
================

NVIDIA Release ${NVIDIA_TENSORFLOW_VERSION} (build ${NVIDIA_BUILD_ID})

Container image Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
Copyright 2017 The TensorFlow Authors.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying project or file.
EOF

if [[ "$(find /usr -name libcuda.so.1) " == " " || "$(ls /dev/nvidiactl) " == " " ]]; then
  echo
  echo "WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available."
  echo "   Use 'nvidia-docker run' to start this container; see"
  echo "   https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker ."
else
  ( /usr/local/bin/checkSMVER.sh )  
  DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version)
  if [[ ! "$DRIVER_VERSION" =~ ^[0-9]*.[0-9]*$ ]]; then
    echo "Failed to detect NVIDIA driver version."
  elif [[ "${DRIVER_VERSION%.*}" -lt "410" ]]; then
    echo "Legacy NVIDIA Driver detected.  ${_CUDA_COMPAT_STATUS}"
  fi
fi

if [[ "$(df -k /dev/shm |grep ^shm |awk '{print $2}') " == "65536 " ]]; then
  echo
  echo "NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be"
  echo "   insufficient for TensorFlow.  NVIDIA recommends the use of the following flags:"
  echo "   nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 ..."
fi

echo

if [[ $# -eq 0 ]]; then
  exec "/bin/bash"
else
  exec "$@"
fi
