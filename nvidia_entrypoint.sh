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

if [[ "$(find /usr -name libcuda.so.1) " == " " || "$(ls /dev/nvidiactl 2>/dev/null) " == " " ]]; then
  echo
  echo "WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available."
  echo "   Use 'nvidia-docker run' to start this container; see"
  echo "   https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker ."
else
  ( /usr/local/bin/checkSMVER.sh )  
  DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version)
  if [[ ! "$DRIVER_VERSION" =~ ^[0-9]*.[0-9]*$ ]]; then
    echo "Failed to detect NVIDIA driver version."
  elif [[ "${DRIVER_VERSION%.*}" -lt "${CUDA_DRIVER_VERSION%.*}" ]]; then
    if [[ "${_CUDA_COMPAT_STATUS}" == "CUDA Driver OK" ]]; then
      echo
      echo "NOTE: Legacy NVIDIA Driver detected.  Compatibility mode ENABLED."
    else
      echo
      echo "ERROR: This container was built for NVIDIA Driver Release ${CUDA_DRIVER_VERSION%.*} or later, but"
      echo "       version ${DRIVER_VERSION} was detected and compatibility mode is UNAVAILABLE."
      echo
      echo "       [[${_CUDA_COMPAT_STATUS}]]"
      sleep 2
    fi
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
