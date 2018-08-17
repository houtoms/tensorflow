#!/bin/bash

set -e

# These tests were chosen based on their runtime (roughly sorted shortest to
# longest) plus a few manual exclusions due to known failure or dependence
# on dynamic libraries only available through bazel.
tests="broadcast_to_ops_test.py
control_flow_util_test.py
ctc_loss_op_test.py
decode_compressed_op_test.py
decode_jpeg_op_test.py
reduce_benchmark_test.py
regex_full_match_op_test.py
string_to_number_op_test.py
tensor_priority_test.py
accumulate_n_eager_test.py
as_string_op_test.py
attention_ops_test.py
bcast_ops_test.py
bincount_op_test.py
bitcast_op_test.py
bucketize_op_test.py
candidate_sampler_ops_test.py
constant_op_eager_test.py
ctc_decoder_ops_test.py
decode_bmp_op_test.py
decode_csv_op_test.py
decode_image_op_test.py
decode_png_op_test.py
decode_raw_op_test.py
denormal_test.py
dense_update_ops_no_tsan_test.py
dense_update_ops_test.py
determinant_op_test.py
draw_bounding_box_op_test.py
edit_distance_op_test.py
extract_image_patches_op_test.py
identity_n_op_py_test.py
identity_op_py_test.py
in_topk_op_test.py
logging_ops_test.py
numerics_test.py
regex_replace_op_test.py
reverse_sequence_op_test.py
save_restore_ops_test.py
softplus_op_test.py
softsign_op_test.py
sparse_concat_op_test.py
sparse_reorder_op_test.py
sparse_reshape_op_test.py
sparse_serialization_ops_test.py
sparse_slice_op_test.py
sparse_to_dense_op_py_test.py
sparse_xent_op_test.py
sparsemask_op_test.py
stack_ops_test.py
string_join_op_test.py
string_split_op_test.py
string_strip_op_test.py
string_to_hash_bucket_op_test.py
substr_op_test.py
summary_audio_op_test.py
summary_image_op_test.py
summary_ops_test.py
summary_tensor_op_test.py
accumulate_n_test.py
aggregate_ops_test.py
argmax_op_test.py
base64_ops_test.py
basic_gpu_test.py
batchtospace_op_test.py
benchmark_test.py
betainc_op_test.py
checkpoint_ops_test.py
clip_ops_test.py
compare_and_bitpack_op_test.py
confusion_matrix_test.py
conv1d_test.py
conv2d_transpose_test.py
cross_grad_test.py
division_future_test.py
division_past_test.py
dynamic_stitch_op_test.py
garbage_collection_test.py
gradient_correctness_test.py
large_concat_op_test.py
manip_ops_test.py
matrix_triangular_solve_op_test.py
relu_op_test.py
reshape_op_test.py
rnn_test.py
slice_op_test.py
softmax_op_test.py
spacetodepth_op_test.py
sparse_add_op_test.py
sparse_split_op_test.py
sparse_tensor_dense_matmul_grad_test.py
sparse_tensors_map_ops_test.py
trace_op_test.py
unique_op_test.py
variable_ops_test.py
xent_op_test.py
zero_division_test.py
cond_v2_test.py
constant_op_test.py
gather_nd_op_test.py
io_ops_test.py
lrn_op_test.py
matrix_logarithm_op_test.py
morphological_ops_test.py
reduce_join_op_test.py
scalar_test.py
session_ops_test.py
sparse_cross_op_test.py
template_test.py
topk_op_test.py
cast_op_test.py
check_ops_test.py
conv3d_backprop_filter_v2_grad_test.py
list_ops_test.py
matrix_exponential_op_test.py
neon_depthwise_conv_op_test.py
parameterized_truncated_normal_op_test.py
parse_single_example_op_test.py
parsing_ops_test.py
spacetobatch_op_test.py
stage_op_test.py
variables_test.py
matrix_solve_op_test.py
nth_element_op_test.py
sparse_tensor_dense_matmul_op_test.py
stack_op_test.py
weights_broadcast_test.py
where_op_test.py
bias_op_test.py
conv3d_transpose_test.py
partitioned_variables_test.py
reader_ops_test.py
resource_variable_ops_test.py
fifo_queue_test.py
fractional_avg_pool_op_test.py
listdiff_op_test.py
lookup_ops_test.py
matrix_inverse_op_test.py
split_op_test.py
tensor_array_ops_test.py
unstack_op_test.py
variable_scope_test.py
diag_op_test.py
fractional_max_pool_op_test.py
gather_op_test.py
map_stage_op_test.py
one_hot_op_test.py
atrous_convolution_test.py
functional_ops_test.py
pad_op_test.py
sets_test.py
dct_ops_test.py
shape_ops_test.py
control_flow_ops_py_test.py
scatter_nd_ops_test.py
cholesky_op_test.py
matrix_solve_ls_op_test.py
padding_fifo_queue_test.py
py_func_test.py
sparse_conditional_accumulator_test.py
dynamic_partition_op_test.py
sparse_matmul_op_test.py
conditional_accumulator_test.py
priority_queue_test.py
scan_ops_test.py
segment_reduction_ops_test.py
barrier_ops_test.py
inplace_ops_test.py
record_input_test.py
embedding_ops_test.py
conv2d_backprop_filter_grad_test.py
batch_matmul_op_test.py"

# Install test dependencies
pip install portpicker

pushd /opt/tensorflow

test_root="tensorflow/python/kernel_tests"
tmp_logfile="/tmp/tf_python_kernel_test.log"

test_count=0
retval=0
set +e
for test_script in $tests; do
    echo "Running $test_script"
    if ! python $test_root/$test_script >& $tmp_logfile ; then
        cat $tmp_logfile
        retval=$(expr $retval + 1)
    fi
    test_count=$(expr $test_count + 1)
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
