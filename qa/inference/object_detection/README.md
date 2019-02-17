# Object detection for QA in CI

This directory is where we maintain scripts and config files needed to
test object detection in the CI. Currently it's used for inference only.

The config files are in 3 sub-directories as follows:

### [generic_acc](tests/generic_acc)

This includes the config files that specify the baseline accuracy.
These are the main config files used in the CI.
Assertions are used to check accuracy (mAP) of the model.

### [no_assertions](tests/no_assertions)

The config files here are the same as in [generic_acc](tests/generic_acc)
except that they don't include any assertions, and thus not used in the CI.


### [xavier_acc_perf](tests/xavier_acc_perf)

The config files here are the same as in [generic_acc](tests/generic_acc)
except that they also include assertions for performance (throughput
and latency), and thus only used in the Xavier pipeline.

