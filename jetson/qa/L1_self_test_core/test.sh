#!/bin/bash

CURRENT_DIR=`pwd`

#Script directory is tensorflow/jetson/qa/L1_self_test_core/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR


#Make sure tensorflow is properly configured
cd ../../
sudo bash auto_conf.sh
cd $SCRIPT_DIR


#Run core tests as they are executed in qa/L1_self_test/test.sh
bazel test  --config=cuda -c opt --verbose_failures --local_test_jobs=1 \
		--test_tag_filters=-no_gpu,-benchmark-test --cache_test_results=no \
		--build_tests_only \
		-- \
		//tensorflow/core/debug:grpc_session_debug_test \
		//tensorflow/core/distributed_runtime/rpc:grpc_session_test_gpu \
		//tensorflow/core/distributed_runtime:cluster_function_library_runtime_test \
		//tensorflow/core/platform/cloud:ram_file_block_cache_test \
	| tee testresult.tmp 
{ grep "test\.log" testresult.tmp || true; } | ../../../qa/show_testlogs 
cd $CURRENT_DIR
