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
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'dagma'
COLUMNS = ('E', 'S', 'C', 'EG', 'DF', 'PGA', 'DR', 'NDWI', 'NDVI', 'LU', 'P', 'PM', 'T', 'TM', 'SD', 'LS')
ALGORITHM = 'dagma'
PARAMETERS = {
    'seed': 42,
    'threads': 1,
    'dagma_l1': 0.03,
    'dagma_stages': 5,
    'dagma_warm_iterations': 30000,
    'dagma_iterations': 60000,
}


def learn_structure(data, args):
    from dagma.linear import DagmaLinear
    model = DagmaLinear(loss_type='l2', verbose=False)
    weights = model.fit(data.copy(), lambda1=args.dagma_l1, w_threshold=0.0,
                        T=args.dagma_stages, warm_iter=args.dagma_warm_iterations,
                        max_iter=args.dagma_iterations, exclude_edges=tuple((i, i) for i in range(len(COLUMNS))))
    details = {'implementation': 'dagma.linear.DagmaLinear', 'loss_type': 'l2',
               'lambda1': args.dagma_l1, 'T': args.dagma_stages,
               'warm_iter': args.dagma_warm_iterations, 'max_iter': args.dagma_iterations,
               'native_h': float(model.h_final)}
    return np.asarray(weights, dtype=np.float64).copy(), details


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
