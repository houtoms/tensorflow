#!/bin/bash

set -e

# Searches for a string (key) in an array of strings
#   1st-arg: key
#   2nd-arg: array of strings
function containsElement {
  local key=$1
  shift
  local list=("$@")
  for l in ${list[@]}; do
    if [[ "$l" == "$key" ]]; then
      echo 1
      return
    fi
  done
  echo 0
}

# Stores all TF-TRT test scripts in array tests
test_root="tensorflow/contrib/tensorrt/"
pushd ../../$test_root
tests="$(find . -type f -name '*_test.py')"

echo tests ${tests[@]}

# Array that specifies which test scripts should not be run.
# We ignore test scripts if running them has no benefit which
# is mostly because of bugs in test scripts.
# Ensure to prepend each test string with ./
#
#./custom_plugin_examples/plugin_test.py
#    It's currently broken. We don't use plugins yet. Once we
#    start using plugins from the registry, then we should
#    write some tests.
#./test/quantization_mnist_test.py
#    It's ignored because it uses data from testdata directory
#    and it's too complicated to set that path of that dir
#    such that it works for both bazel and python.
ignored_tests="
./custom_plugin_examples/plugin_test.py
./test/quantization_mnist_test.py
"

tmp_logfile="/tmp/tf_trt_test.log"
test_count=0
retval=0
set +e
for test_script in $tests; do
    found=$(containsElement $test_script ${ignored_tests[@]})
    if [ $found = "1" ]; then
        echo "Ignoring $test_script"
    else
      echo "Running $test_script"
      if ! python -u $test_script >& $tmp_logfile ; then
          cat $tmp_logfile
          retval=$(expr $retval + 1)
      fi
      test_count=$(expr $test_count + 1)
    fi
done
set -e

rm -f $tmp_logfile

popd

if [ "$retval" == "0" ] ; then
    echo "All $test_count tests PASSED"
else
    echo "$retval / $test_count tests FAILED"
fi

exit $retval
