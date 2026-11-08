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
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'dag_gnn'
COLUMNS = ('E', 'S', 'C', 'EG', 'DF', 'PGA', 'DR', 'NDWI', 'NDVI', 'LU', 'P', 'PM', 'T', 'TM', 'SD', 'LS')
ALGORITHM = 'dag_gnn'
PARAMETERS = {
    'seed': 42,
    'threads': 1,
    'gnn_epochs': 200,
    'gnn_iterations': 10,
    'gnn_l1': 0.0,
    'hidden': 32,
    'batch_size': 512,
}


def learn_structure(data, args):
    import torch
    from castle.algorithms import DAG_GNN
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = DAG_GNN(encoder_type='mlp', decoder_type='mlp', encoder_hidden=args.hidden,
                    decoder_hidden=args.hidden, epochs=args.gnn_epochs,
                    k_max_iter=args.gnn_iterations, batch_size=min(args.batch_size, len(data)),
                    tau_a=args.gnn_l1, graph_threshold=0.0,
                    h_tolerance=1e-8, seed=args.seed, device_type='cpu')
    model.learn(data.copy(), columns=list(COLUMNS))
    model.encoder.eval()
    with torch.no_grad():
        _, learned = model.encoder(torch.from_numpy(data[:1].copy()).double().unsqueeze(-1))
    details = {'implementation': 'gcastle.DAG_GNN', 'encoder': 'mlp', 'decoder': 'mlp',
               'epochs': args.gnn_epochs, 'k_max_iter': args.gnn_iterations,
               'hidden': args.hidden, 'batch_size': min(args.batch_size, len(data)),
               'tau_a': args.gnn_l1, 'device': 'cpu'}
    return learned.detach().cpu().numpy().copy(), details


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
