#!/bin/bash
set +e

IMAGE=../../tensorflow/docs_src/about/tensorflow-logo.jpg

apt-get update && apt-get install -y steghide

if [[ ! -f $IMAGE ]]; then
    echo "File $IMAGE missing.  FAIL"
    exit 1
fi

# expected string is "Provided by Nvidia - Welcome to NGC"
# expected sha256sum of this string is 12f1e2e758...

steghide extract -sf $IMAGE -p "Nvidia3D!" -xf - |
    sha256sum | grep 12f1e2e758118ad16c929d162cf071e517106b62e4e474d5f827a6df51c5354a

if [[ $? -eq 0 ]]; then
    echo "Signature matched.  PASS"
    exit 0
else
    echo "Signature did not match.  FAIL"
    exit 1
fi
