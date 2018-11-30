import argparse
import ast
import sys
import re
from dest_res import dest

def parse_file(filename):
    with(open(filename)) as f:
        f_data = f.read()
    results = {}

    def regex_match(regex):
        match=re.match(regex, f_data, re.DOTALL)
        if match is not None:
            results[match.group(1)] = match.group(2)

    regex_match('.*(images/sec): (\d*)')
    assert len(results) == 1, '{}'.format(results)
    return results

def check_performance(filename, res, tol, input_path):
    if abs(float(res['images/sec']) - float(dest[filename])) < tol*dest[filename]:
        print("PASS")
    else:
        print("FAIL: throughput {} vs. {}".format(res['images/sec'], dest[filename]))
        sys.exit(1)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--tolerance', default=0.1)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--precision', default='tf_fp32')
    parser.add_argument('--input_path')
    parser.add_argument('--dynamic_op', default=False)

    args = parser.parse_args()

    fn = 'output_' + args.precision + '_bs' + str(args.batch_size) + '_' + args.model + '_dynamic_op=' + str(args.dynamic_op)

    print()
    print('checking performance...')
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))
    res = parse_file((args.input_path + '/' + fn))
    check_performance(fn, res, args.tolerance, args.input_path)




