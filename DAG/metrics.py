from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def read_matrix(path):
    matrix = pd.read_csv(path, index_col=0, encoding='utf-8-sig', float_precision='round_trip')
    if matrix.shape != (16, 16) or not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError('Expected a 16 x 16 matrix with unique labels.')
    if set(matrix.index) != set(matrix.columns):
        raise ValueError('Row and column labels must match.')
    matrix = matrix.loc[matrix.index, matrix.index]
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError('Matrix contains nonfinite values.')
    return matrix


def graph_from_matrix(path):
    return nx.from_pandas_adjacency(read_matrix(path), create_using=nx.DiGraph)


def node_metrics(path):
    graph = graph_from_matrix(path)
    return pd.DataFrame([{'field': node, 'in_degree': graph.in_degree(node), 'out_degree': graph.out_degree(node),
                          'ancestors': len(nx.ancestors(graph, node)), 'descendants': len(nx.descendants(graph, node)),
                          'is_root': graph.in_degree(node) == 0, 'is_leaf': graph.out_degree(node) == 0}
                         for node in graph])


def pathway_table(path, target='LS', max_paths=10000):
    graph = graph_from_matrix(path)
    if target not in graph:
        raise ValueError('Unknown target: ' + target)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('Pathway analysis requires an acyclic graph.')
    if not isinstance(max_paths, int) or max_paths < 1:
        raise ValueError('max_paths must be a positive integer.')
    rows = []
    for source in graph:
        if source == target or graph.in_degree(source) != 0:
            continue
        for nodes in nx.all_simple_paths(graph, source, target):
            if len(rows) >= max_paths:
                raise ValueError('Number of pathways exceeds max_paths.')
            weights = np.array([graph[a][b]['weight'] for a, b in zip(nodes[:-1], nodes[1:])])
            rows.append({'source': source, 'target': target, 'pathway': ' -> '.join(nodes),
                         'length': len(weights), 'weight_product': float(np.prod(weights))})
    return pd.DataFrame(rows, columns=['source', 'target', 'pathway', 'length', 'weight_product'])


def compare_graphs(reference, prediction):
    ref, pred = graph_from_matrix(reference), graph_from_matrix(prediction)
    if set(ref) != set(pred):
        raise ValueError('Graphs must contain the same fields.')
    a, b = set(ref.edges()), set(pred.edges())
    sk_a, sk_b = {frozenset(edge) for edge in a}, {frozenset(edge) for edge in b}
    common = len(a & b)
    precision = common / len(b) if b else 0.0
    recall = common / len(a) if a else 0.0
    reversals = sum((j, i) in b - a for i, j in a - b if i != j)
    return {'reference_edges': len(a), 'prediction_edges': len(b), 'common_edges': common,
            'added_edges': len(b - a), 'removed_edges': len(a - b), 'reversals': reversals,
            'shd_reversal_cost_one': len(a - b) + len(b - a) - reversals,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'edge_jaccard': common / len(a | b) if a | b else 1.0,
            'skeleton_jaccard': len(sk_a & sk_b) / len(sk_a | sk_b) if sk_a | sk_b else 1.0}


def compare_outputs(folder):
    folder = Path(folder)
    available = [name for name in ('lingam', 'dagma', 'notears', 'dag_gnn') if (folder / name / 'weights.csv').is_file()]
    rows = []
    for a, b in combinations(available, 2):
        rows.append({'algorithm_a': a, 'algorithm_b': b,
                     **compare_graphs(folder / a / 'weights.csv', folder / b / 'weights.csv')})
    return pd.DataFrame(rows)
