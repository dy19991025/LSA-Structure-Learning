import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def save_results(folder, weights, columns, algorithm, samples, options, details, elapsed, caught):
    folder = Path(folder)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(columns), len(columns)) or not np.isfinite(weights).all():
        raise ValueError('The estimator returned invalid weights.')
    folder.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(weights, index=columns, columns=columns).rename_axis('source')
    matrix.to_csv(folder / 'weights.csv', encoding='utf-8-sig', float_format='%.17g')
    adjacency = (matrix != 0).astype(np.int32)
    adjacency.to_csv(folder / 'adjacency.csv', encoding='utf-8-sig')
    rows = [{'source': columns[i], 'target': columns[j], 'weight': float(weights[i, j])}
            for i, j in zip(*np.nonzero(weights))]
    pd.DataFrame(rows, columns=['source', 'target', 'weight']).to_csv(
        folder / 'edges.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    graph = nx.from_pandas_adjacency(matrix, create_using=nx.DiGraph)
    nx.write_graphml(graph, folder / 'graph.graphml')
    acyclic = nx.is_directed_acyclic_graph(graph)
    summary = {'algorithm': algorithm, 'samples': samples, 'fields': list(columns),
               'edges': len(rows), 'self_loops': nx.number_of_selfloops(graph), 'is_dag': acyclic,
               'topological_order': list(nx.topological_sort(graph)) if acyclic else None,
               'elapsed_seconds': elapsed, 'parameters': options, 'implementation': details,
               'warnings': sorted(set(str(item.message) for item in caught))}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    return summary
