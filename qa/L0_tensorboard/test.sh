#!/bin/bash
set +e
set +x

GPUS=$(nvidia-smi -L 2>/dev/null| wc -l || echo 1)
[[ $GPUS -gt 4 ]] && GPUS=4
export CUDA_VISIBLE_DEVICES=$(seq -s',' 0 $((GPUS-1)))
( cmdpid=$BASHPID; (sleep 20; kill -s SIGINT $cmdpid) & exec tensorboard --logdir /tmp )
RET=$?
if [[ $RET -eq 0 ]]; then
    echo Test succeeded
else
    echo Test failed
fi
exit $RET
