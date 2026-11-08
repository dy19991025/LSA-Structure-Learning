from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .metrics import graph_from_matrix, read_matrix

STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
         'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42}


def save_figure(figure, output_prefix):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for extension in ('png', 'svg', 'pdf'):
            path = Path(str(prefix) + '.' + extension)
            figure.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            paths.append(path)
    finally:
        plt.close(figure)
    return paths


def plot_network(matrix_path, output_prefix):
    graph = graph_from_matrix(matrix_path)
    positions = nx.circular_layout(graph)
    colours = ['#2166AC' if data['weight'] >= 0 else '#B35806' for _, _, data in graph.edges(data=True)]
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(9, 9))
        nx.draw_networkx_nodes(graph, positions, ax=axis, node_size=1100,
                               node_color=['#FED7AA' if node == 'LS' else '#DBEAFE' for node in graph],
                               edgecolors='#334155', linewidths=0.9)
        nx.draw_networkx_labels(graph, positions, ax=axis, font_size=10)
        nx.draw_networkx_edges(graph, positions, ax=axis, node_size=1100,
                               edge_color=colours, arrowsize=11, width=0.8, alpha=0.55,
                               connectionstyle='arc3,rad=0.08')
        from matplotlib.lines import Line2D
        axis.legend(handles=[Line2D([0], [0], color='#2166AC', label='Positive weight'),
                             Line2D([0], [0], color='#B35806', label='Negative weight')],
                    loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
        axis.margins(0.15)
        axis.set_axis_off()
        return save_figure(figure, output_prefix)


def plot_weights(matrix_path, output_prefix):
    matrix = read_matrix(matrix_path)
    values = matrix.to_numpy(dtype=float)
    limit = max(float(np.max(np.abs(values))), 1e-12)
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(8, 7))
        heatmap = axis.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        axis.set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha='right')
        axis.set_yticks(range(len(matrix)), matrix.index)
        axis.set_xlabel('Target')
        axis.set_ylabel('Source')
        figure.colorbar(heatmap, ax=axis, label='Raw weight', shrink=0.8)
        return save_figure(figure, output_prefix)
