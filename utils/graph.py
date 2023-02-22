
import numpy as np

def generate_graph(N, p=0.2, seed=123):
    nodes = list(range(N))
    edges = set()
    for n, n_ in zip(nodes, nodes[1:]+[0]):
        edges.add( (n, n_) )

    rng = np.random.default_rng(seed=seed)
    for n in nodes:
        for n_ in nodes:
            if n!=n_ and rng.random() < p:
                edges.add( (n, n_))

    return set(nodes), edges

def generate_weighted_graph(N, p=0.2, edgeseed=123, weightseed=321):
    rng = np.random.default_rng(seed=weightseed)
    nodes, edges = generate_graph(N, p, edgeseed)
    weights = dict()
    for e in edges:
        if not ( e[1], e[0] ) in weights: # bi direction should be the same weight
            weights[e] = rng.random() * 100 // 1 /100
        else :
            weights[e] = weights[(e[1], e[0])]

    return nodes, weights
