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

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))

if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)

from collections import OrderedDict
import json

import horovod.tensorflow as hvd

from main import *
from utils import *


class JsonBenchLogger(BenchLogger):

    def __init__(self, name, total_bs, warmup_iter, save_raport_fn):
        super(JsonBenchLogger, self).__init__(name, total_bs, warmup_iter)
        self.save_raport = save_raport_fn

    def end_callback(self):
        if hvd.rank() == 0:
            super(JsonBenchLogger, self).end_callback()
            metrics = OrderedDict(
                [('batch_time', self.batch_time.avg), ('total_ips', self.total_bs / (self.batch_time.avg))]
            )

            self.save_raport(metrics)


def save_raport(filename):

    def _save_raport(metrics):
        raport = OrderedDict([
            ('model', flags.arch),
            ('ngpus', hvd.size()),
            ('cmd', sys.argv),
            ('metrics', metrics),
        ])
        with open(filename, 'w') as f:
            json.dump(raport, f, indent=4)

    return _save_raport


def getJsonBenchLogger(save_raport_fn):

    def _const(name, total_bs, warmup_iter):
        return JsonBenchLogger(name, total_bs, warmup_iter, save_raport_fn)

    return _const


def main(args, flags):
    if flags.trainbench or flags.inferbench:
        logger = getJsonBenchLogger(save_raport(flags.raport_file))
    train_net(args, flags, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorFlow ImageNet Testing Suite')

    add_parser_arguments(parser)
    parser.add_argument(
        '--raport-file',
        default='experiment_raport.json',
        type=str,
        help='file in which to store JSON experiment raport'
    )
    args, flags = utils.parse_cmdline(resnet.default_args, parser)

    main(args, flags)
