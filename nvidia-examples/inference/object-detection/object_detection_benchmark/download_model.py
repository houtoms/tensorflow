import argparse
from .models import MODELS
import subprocess
import os


def download_model(name, output_dir='models'):
    global MODELS
    model = MODELS[name]
    subprocess.call(['mkdir', '-p', output_dir])
    tar_file = os.path.join(output_dir, os.path.basename(model.url))

    if not os.path.exists(os.path.join(output_dir, model.extract_dir)):
        subprocess.call(['wget', model.url, '-O', tar_file])
        subprocess.call(['tar', '-xzf', tar_file, '-C', output_dir])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output_dir', default='models')
    args = parser.parse_args()

    download_model(args.model, output_dir=args.output_dir)
