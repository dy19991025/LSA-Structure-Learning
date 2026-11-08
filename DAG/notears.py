import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import warnings

os.environ['CASTLE_BACKEND'] = 'pytorch'
if __package__ in (None, ''):
    sys.path = [entry for entry in sys.path if Path(entry or '.').resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from DAG.io import save_results

INPUT_FILE = Path(r'D:\Landslide\data\landslide_data.csv')
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'notears'
COLUMNS = ('E', 'S', 'C', 'EG', 'DF', 'PGA', 'DR', 'NDWI', 'NDVI', 'LU', 'P', 'PM', 'T', 'TM', 'SD', 'LS')
ALGORITHM = 'notears'
PARAMETERS = {
    'seed': 42,
    'threads': 1,
    'notears_l1': 0.1,
    'notears_iterations': 100,
}


def learn_structure(data, args):
    from castle.algorithms import Notears
    model = Notears(lambda1=args.notears_l1, loss_type='l2', max_iter=args.notears_iterations,
                    h_tol=1e-8, w_threshold=0.0)
    model.learn(data.copy(), columns=list(COLUMNS))
    details = {'implementation': 'gcastle.Notears', 'loss_type': 'l2',
               'lambda1': args.notears_l1, 'max_iter': args.notears_iterations,
               'acyclicity': 'trace((I + W*W/d)^d) - d'}
    return np.asarray(model.weight_causal_matrix, dtype=np.float64).copy(), details


def run(input_file=INPUT_FILE, output_dir=OUTPUT_DIR, **parameters):
    unknown = set(parameters) - set(PARAMETERS)
    if unknown:
        raise ValueError('Unknown parameters: ' + ', '.join(sorted(unknown)))
    options = dict(PARAMETERS, **parameters)
    args = argparse.Namespace(**options)
    frame = pd.read_csv(input_file, encoding='utf-8-sig')
    data = frame.loc[:, list(COLUMNS)].to_numpy(dtype=np.float64)
    if len(data) < 2 or not np.isfinite(data).all():
        raise ValueError('Input requires at least two rows and finite numeric values.')
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError('Use a new or empty output directory: ' + str(output_dir))
    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.perf_counter()
    with threadpool_limits(limits=args.threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        weights, details = learn_structure(data, args)
    elapsed = time.perf_counter() - start
    return save_results(output_dir, weights, COLUMNS, ALGORITHM, len(data), options,
                        details, elapsed, caught)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=INPUT_FILE)
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR)
    for key, value in PARAMETERS.items():
        parser.add_argument('--' + key.replace('_', '-'), type=type(value), default=value)
    arguments = parser.parse_args()
    values = {key: getattr(arguments, key) for key in PARAMETERS}
    result = run(arguments.input, arguments.output, **values)
    print(json.dumps(result, indent=2, allow_nan=False))
