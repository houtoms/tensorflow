# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import json
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _add_bool_argument(parser, name=None, default=False, required=False, help=None):

    if not isinstance(default, bool):
        raise ValueError()

    feature_parser = parser.add_mutually_exclusive_group(required=required)

    feature_parser.add_argument('--' + name, dest=name, action='store_true', help=help, default=default)
    feature_parser.add_argument('--no' + name, dest=name, action='store_false')
    feature_parser.set_defaults(name=default)


def _parse_cmdline():

    parser = argparse.ArgumentParser(description='Tensorflow UNet Industrial Benchmark Tests')

    _add_bool_argument(
        parser=parser,
        name="use_tf_amp",
        default=False,
        required=False,
        help="Enable Automatic Mixed Precision to speedup FP32 computation using tensor cores"
    )

    _add_bool_argument(
        parser=parser,
        name="use_xla",
        default=False,
        required=False,
        help="Enable Tensorflow XLA to maximise performance."
    )

    FLAGS, unknown_args = parser.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    return FLAGS


def benchmark(use_tf_amp, use_xla):

    failed_tests = 0
    passed_tests = 0

    if use_xla:
        raise ValueError("XLA is currently not supported")

    amp_flag = "--use_tf_amp" if use_tf_amp else "--nouse_tf_amp"
    xla_flag = "--use_xla" if use_xla else "--nouse_xla"

    base_dir = os.path.dirname(os.path.realpath(__file__))
    baseline_dir = os.path.join(base_dir, "benchmark_baselines/inferbench")

    performance_file = "/tmp/performances_eval.json"

    unet_dir = os.path.join(base_dir, "../../../../nvidia-examples/UNet_Industrial")

    for class_id in range(1, 11):

        front_cmd = "rm -rf /tmp && pip install %s/dllogger && \\" % unet_dir

        exec_command = front_cmd + """
            python %s/main.py \\
                    --unet_variant='tinyUNet' \\
                    --activation_fn='relu' \\
                    --exec_mode='inference_benchmark' \\
                    --iter_unit='batch' \\
                    --num_iter=1500 \\
                    --batch_size=16 \\
                    --warmup_step=500 \\
                    --results_dir="/tmp/" \\
                    --data_dir="/data/dagm2007" \\
                    --dataset_name='DAGM2007' \\
                    --dataset_classID="%d" \\
                    --data_format='NCHW' \\
                    --use_auto_loss_scaling \\
                    %s \\
                    %s \\
                    --learning_rate=1e-4 \\
                    --learning_rate_decay_factor=0.8 \\
                    --learning_rate_decay_steps=500 \\
                    --rmsprop_decay=0.9 \\
                    --rmsprop_momentum=0.8 \\
                    --loss_fn_name='adaptive_loss' \\
                    --weight_decay=1e-5 \\
                    --weight_init_method='he_uniform' \\
                    --augment_data \\
                    --display_every=250 \\
                    --debug_verbosity=0
        """ % (unet_dir, class_id, amp_flag, xla_flag)

        print(
            "[Running] QA Job - UNet TF Industrial - Inference Benchmark\n"
            "\t[*] Use TF AMP: %s\n"
            "\t[*] Use XLA: %s\n"
            "\t[*] DAGM Class ID: %d" %
            (use_tf_amp, use_xla, class_id)
        )

        exit_code = subprocess.call(exec_command, shell=True, stdout=subprocess.DEVNULL)

        if exit_code != 0:
            raise RuntimeError("CMD: \"{}\" \n exited with status {}".format(exec_command, exit_code))

        baseline_file = os.path.join(
            baseline_dir,
            "UNet_Industrial_1GPU_%s.json" % ("AMP" if use_tf_amp else "FP32")
        )

        try:

            with open(baseline_file) as baseline_json_file:
                baseline_dict = json.loads(baseline_json_file.read())

            with open(performance_file) as results_json_file:
                results_dict = json.loads(results_json_file.read())

            if (
                    float(results_dict["throughput"]) < float(results_dict["throughput"]) or
                    float(results_dict["processing_time"]) > float(results_dict["processing_time"])
            ):

                raise RuntimeError(
                    "[Error] The model did not performed to the target baseline.\n"
                    "HParams:\n"
                    "\t[*] Use TF AMP: %s\n"
                    "\t[*] Use XLA: %s\n"
                    "\t[*] DAGM Class ID: %02d\n\n"
                    "Baseline: %s\n\n"
                    "Results: %s\n\n" % (
                        use_tf_amp,
                        use_xla,
                        class_id,
                        baseline_dict,
                        results_dict
                    )
                )

        except RuntimeError as e:
            sys.stderr.write(str(e))
            failed_tests += 1
            continue

        else:
            passed_tests += 1
            print("=> Job ended successfully")

        print("\n##########################################################\n")

    return passed_tests, failed_tests


if __name__ == "__main__":

    FLAGS = _parse_cmdline()

    successes, failures = benchmark(use_tf_amp=FLAGS.use_tf_amp, use_xla=FLAGS.use_xla)

    print("[Summary] - Success: %d - Failures: %d" % (successes, failures))
    exit(failures)
