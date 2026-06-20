from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def correlation_distance_matrix(returns_window: pd.DataFrame) -> pd.DataFrame:
    corr = returns_window.corr(method="pearson").clip(-1.0, 1.0)
    dist = np.sqrt(2 * (1 - corr))
    np.fill_diagonal(dist.values, 0.0)
    return dist


def build_weighted_graph_from_distance(distance_matrix: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    assets = list(distance_matrix.columns)
    g.add_nodes_from(assets)
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            b = assets[j]
            d = float(distance_matrix.loc[a, b])
            sim = 1.0 / (1.0 + d)
            g.add_edge(a, b, weight=sim)
    return g


def louvain_partition(graph: nx.Graph, seed: int | None = None) -> list[set[str]]:
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return [{n} for n in graph.nodes]
    communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    return [set(c) for c in communities]


def consensus_similarity_matrix(
    partitions: list[list[set[str]]],
    assets: list[str],
) -> np.ndarray:
    n = len(assets)
    idx = {a: i for i, a in enumerate(assets)}
    sim = np.zeros((n, n), dtype=float)
    if not partitions:
        np.fill_diagonal(sim, 1.0)
        return sim
    for part in partitions:
        for community in part:
            members = [idx[m] for m in community if m in idx]
            for i in members:
                sim[i, i] += 1
            for p in range(len(members)):
                for q in range(p + 1, len(members)):
                    i, j = members[p], members[q]
                    sim[i, j] += 1
                    sim[j, i] += 1
    sim /= float(len(partitions))
    np.fill_diagonal(sim, 1.0)
    return sim


def stable_clusters_from_similarity(
    similarity: np.ndarray, assets: list[str], threshold: float = 0.5
) -> list[list[str]]:
    if similarity.shape[0] != len(assets):
        raise ValueError("Similarity matrix and assets length mismatch")
    g = nx.Graph()
    g.add_nodes_from(assets)
    n = len(assets)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                g.add_edge(assets[i], assets[j], weight=float(similarity[i, j]))
    return [sorted(list(c)) for c in nx.connected_components(g)]
