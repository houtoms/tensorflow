import argparse
import ast
import sys
import re

def parse_log_file(filename):
    with open(filename) as f:
        f_data = f.read()
    results = {}
    def regex_match(key, regex):
        match = re.match(regex, f_data, re.DOTALL)
        if match is not None:
            results[key] = match.group(2)
    regex_match('objects_per_sec','.*(\*\*\* Avg objects per second): (\d*\.\d*)')
    regex_match('latency_mean', '.*(\*\*\* Avg time per step): (\d*\.\d*)s')
    assert len(results) == 2, '{}'.format(results)
    # Convert latency from s to ms
    results['latency_mean'] = 1000.0 * float(results['latency_mean'])
    return results

def parse_results_file(filename):
    with open(filename) as f:
        f_data = f.read()
    result = f_data.split(' ')[2]
    return float(result)

def check_score(bleu_score, model, tol):
    dest = {
        'transformer_config.py': 26.40,
        'trt_transformer_config.py': 26.40,
        'convs2s_config.py': 25.00,
        'trt_convs2s_config.py': 25.00,
    }
    if abs(bleu_score - dest[model]) < tol:
        print("PASS")
    else:
        print("FAIL: BLEU score {} vs. {}".format(res, dest[model]))
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_result')
    parser.add_argument('--input_log')
    parser.add_argument('--model')
    parser.add_argument('--tolerance', type=float, default=0.1)
    args = parser.parse_args()

    print()
    print('results for {}'.format(args.model))
    results_to_print = vars(args)
    results_to_print.update(parse_log_file(args.input_log))
    bleu_score = parse_results_file(args.input_result)
    results_to_print['bleu_score'] = bleu_score
    for k, v in sorted(results_to_print.items()):
        print('{}: {}'.format(k, '%.2f'%v if type(v)==float else v))
    
    print('checking BLEU score...')
    check_score(bleu_score, args.model, args.tolerance)
