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

import argparse
import json
import sys
import subprocess
import re
import os

from collections import OrderedDict

ACC_THR = 0.99

parser = argparse.ArgumentParser(description='Tesnorflow Benchmark Tests')

parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--ngpus', default=1, type=int)

parser.add_argument(
    '--iterations',
    type=int,
    default=90,
    metavar='N',
    help='Run N iterations while benchmarking (ignored when training and validation)'
)

parser.add_argument('--precision', default='fp32', choices=['fp16', 'fp32'], help='Model precision')

parser.add_argument(
    '--top1-baseline', type=float, metavar='PERCENTAGE', required=True, help='Baseline top-1 accuracy (percentage)'
)
parser.add_argument(
    '--top5-baseline', type=float, metavar='PERCENT', required=True, help='Baseline top-5 accuracy (percentage)'
)

parser.add_argument('--data_dir', default="/data/imagenet", type=str, metavar='<PATH>', help='path to the dataset')
parser.add_argument('--results_dir', default="/results", type=str, metavar='<PATH>', help='path to the results')

args = parser.parse_args()

command = "{{}} main.py --mode=train_and_evaluate --num_iter={num_iter} --warmup_steps=100 --precision={precision} --iter_unit=epoch --data_dir={data_dir} --batch_size={batch_size} --results_dir={results_dir}/{exp_name}"

METRICS = {'top1': 'Top-1 Accuracy:\s*(.+)\s*$', 'top5': 'Top-5 Accuracy:\s*(.+)\s*$'}


def parse_log(data):
    results = {}

    for metric, pattern in METRICS.items():
        values = []
        matches = re.finditer(pattern, data, re.MULTILINE)

        for m in matches:
            values.append(float(m.group(1)))

        if len(values) != 1:
            print('Error parsing output, could not determine metric: {}'.format(metric))
            assert False

        results[metric] = values[0]

    return results


def benchmark(command, args):
    sgpu = str(sys.executable)
    mgpu = "mpiexec --allow-run-as-root --bind-to socket -np {ngpu} {sgpu}"

    exp_name = "{}GPU_{}BS_training".format(args.ngpus, args.bs)
    results_path = os.path.join(args.results_dir, exp_name)
    os.makedirs(results_path)
    log_path = "{}/stdout.log".format(results_path)

    mgpu_str = mgpu.format(ngpu=args.ngpus, sgpu=sgpu)

    cmd = command.format(
        mode='training',
        precision=args.precision,
        num_iter=args.iterations,
        batch_size=args.bs,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        exp_name=exp_name
    )

    cmd = cmd.format(sgpu if args.ngpus == 1 else mgpu_str)

    print(cmd.split())

    with open(log_path, 'w') as log_file:
        exit_code = subprocess.call(cmd.split(), stdout=log_file)

    if exit_code != 0:
        print("CMD: \"{}\" exited with status {}".format("".join(cmd), exit_code))
        assert False
    else:
        print("Job ended sucessfully")

    with open(log_path, 'r') as log_file:
        log = log_file.read()

    results = parse_log(log)

    for metric, value in results.items():
        print('{}: {}'.format(metric, value))

    return results


def check(results, baseline):
    allright = True
    for m, result in results.items():
        reference = baseline[m]
        if result < ACC_THR * reference:
            allright = False
            print(
                "Metric: {} Result ( {} ) is more than {} times lower than reference ( {} )".format(
                    m, result, ACC_THR, reference
                )
            )

    return allright


baseline = {'top1': args.top1_baseline, 'top5': args.top5_baseline}

results = benchmark(command, args)

if check(results, baseline):
    print("&&&& PASSED")
    exit(0)
else:
    print("&&&& FAILED")
    exit(1)
