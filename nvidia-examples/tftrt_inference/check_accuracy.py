import argparse
import ast
import sys
import re


def parse_file(filename):
    with open(filename) as f:
        d = {}
        m = re.compile('model: \D')
        a = re.compile('accuracy: \d') 
        for line in f:
            if re.match(m, line) or re.match(a, line):
                (key, val) = line.split(': ')
                d[key] = val[:-1]
        assert (len(d) == 2)
        return d


def check_accuracy(res, tol):
    dest = {
        'resnet_v1_50': 0.7590,
        'resnet_v2_50': 0.7606,
        'vgg_16': 0.7089,
        'vgg_19': 0.7100,
        'inception_v3': 0.7798,
        'inception_v4': 0.8019,
        'mobilenet_v2': 0.7408,
        'mobilenet_v1': 0.7101,
        'nasnet_large': 0.8272,
        'nasnet_mobile': 0.7396,
    }
    if abs(float(res['accuracy']) - dest[res['model']]) < tol:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input')
    parser.add_argument('--tolerance', dest='tolerance', type=float, default=0.001)
    
    args = parser.parse_args()
    filename = args.input
    tolerance = args.tolerance
    check_accuracy(parse_file(filename), tolerance)

