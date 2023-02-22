
import os
import pickle
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from dimod.utilities import qubo_to_ising

from sqa.solver import SQASolver
from utils.convert import arr_to_Qdict, spin_to_q

class HCP:
    """Hamiltonian Cycle Problem
    """

    def __init__(self, Vset: set, Eset: set):
        '''
            Args:
                V (set) : the set of nodes
                E (set) : the set of edges. { (i, j), (u, v), (i, u), ...}
        '''
        for e in Eset:
            u, v = e
            if u not in Vset:
                raise AttributeError(f'edge set containts {u}, but {u} is not in the node set')
            if v not in Vset:
                raise AttributeError(f'edge set containts {v}, but {v} is not in the node set')

        self.Vset = Vset
        self.Eset = Eset
        self.listedVset = list(Vset)
        self.listedVset = list(Vset)

        self.N = len(Vset)
        self.qubo_shape = ((self.N-1)**2, (self.N-1)**2)
        self.V = list(range(len(Vset)))
        self.E = [ (self.listedVset.index(e[0]), self.listedVset.index(e[1])) for e in Eset ]

        self.set_node_constraint()
        self.set_order_constraint()
        self.set_edge_constraint()

    def plot_net(self, save_to=''):
        fig,ax = plt.subplots()
        G = nx.DiGraph()
        G.add_nodes_from(self.Vset)
        if hasattr(self, 'Wdict'):
            G.add_weighted_edges_from(list( (*e, w) for e, w in self.Wdict.items() ) )
            layout = partial(nx.spring_layout, iterations=200, seed=123)
        else:
            G.add_edges_from(self.Eset)
            layout = nx.circular_layout

        pos = layout(G)

        if hasattr(self, 'sqa_solution'):
            edges = self.get_edges_from_arr(self.best_sa)
            edgelist = list(edges)
            nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color='blue', width=5, ax=ax)

        nx.draw_networkx(G, pos, edge_color='black', width=1, ax=ax)

        if hasattr(self, 'Wdict'):
            nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, "weight"), ax=ax)
        if save_to:
            plt.savefig(save_to)

    def plot_qubo(self, save_to=''):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        sns.heatmap(self.node_constraint, cmap='bwr', center=0, ax=axes[0,0])
        axes[0, 0].set_title('node constraint')
        sns.heatmap(self.order_constraint, cmap='bwr', center=0, ax=axes[0,1])
        axes[0, 1].set_title('order constraint')
        sns.heatmap(self.edge_constraint, cmap='bwr', center=0, ax=axes[1,0])
        axes[1, 0].set_title('edge constarint')
        if hasattr(self, 'whole_qubo'):
            sns.heatmap(self.whole_qubo, cmap='bwr', center=0, ax=axes[1,1])
        axes[1,1].set_title('whole qubo')
        if save_to:
            plt.savefig(save_to)

    def encode(self, v, j):
        j = j % self.N
        v = v % self.N
        if j == 0:
            raise ValueError('j==0 can not be encoded because (v, j) == (v, 0) is excluded in the qubo expression')
        if v == 0:
            raise ValueError('v==0 can not be encoded because (v, j) == (0, j) is excluded in the qubo expression')

        return v-1 + (self.N-1)*(j-1)

    def decode(self, idx):
        v = idx % (self.N-1)
        j = idx // (self.N-1)
        return v+1, j+1

    def set_node_constraint(self):
        '''for each node, an order must be assigned to that.
        '''
        qubo = np.zeros(self.qubo_shape)
        for v in range(1, self.N):
            for j in range(1, self.N):
                qubo[self.encode(v, j), self.encode(v, j)] -= 2
                for _j in range(1, self.N):
                    qubo[self.encode(v, j), self.encode(v, _j)] += 1

        self.node_constraint = qubo

    def set_order_constraint(self):
        qubo = np.zeros(self.qubo_shape)
        for j in range(1, self.N):
            for v in range(1, self.N):
                qubo[self.encode(v, j), self.encode(v, j)] -= 2
                for _v in range(1, self.N):
                    qubo[self.encode(v, j), self.encode(_v, j)] += 1

        self.order_constraint = qubo

    def set_edge_constraint(self):
        comp_edges = { (u, v) for u in self.V for v in self.V if ( (u, v) not in self.E) and u != v }
        qubo = np.zeros(self.qubo_shape)
        for ce in comp_edges:
            if ce[0] == 0: # start from the node 0
                qubo[self.encode(ce[1], 1), self.encode(ce[1], 1)] += 1
            elif ce[1] == 0: # return to the node 0
                qubo[self.encode(ce[0], self.N-1), self.encode(ce[0], self.N-1)] += 1
            else:
                for j in range(1, self.N-1):
                    qubo[self.encode(ce[0], j), self.encode(ce[1], j+1)] += 1

        self.edge_constraint = qubo

    def set_whole_qubo(self, A, B, C):
        '''set three constraints.
        Args:
            A (float): weight term of the node constraint
            B (float): weight term of the order constraint
            C (float): weight term of the edge constraint
        '''
        self.A = A
        self.B = B
        self.C = C

        self.set_node_constraint()
        self.set_order_constraint()
        self.set_edge_constraint()

        self.whole_qubo = A*self.node_constraint + B*self.order_constraint + C*self.edge_constraint

    def get_edges_from_arr(self, arr):
        order = [set() for n in range(self.N)]
        order[0].add(0)

        for idx, s in enumerate(arr):
            if s == 1:
                v, j = self.decode(idx)
                order[j].add(v)

        edges = set()
        j = 0
        order.append({0})
        for starts, ends in zip(order[:-1], order[1:] ):
            if len(starts) > 0 and len(ends) > 0 :
                edges = edges | { (u, v) for u in starts for v in ends if u!=v }
            j = j + 1

        return edges

    def check_node_constraint(self, arr):
        checker = [1] + [0 for n in range(self.N-1)]
        for idx, s in enumerate(arr):
            if s==1:
                v, j = self.decode(idx)
                if checker[v]==1:
                    return False
                checker[v] += 1
            else:
                continue

        if all([c==1 for c in checker]):
            return True
        else:
            return False

    def check_order_constraint(self, arr):
        checker = [1] + [0 for n in range(self.N-1)]
        for idx, s in enumerate(arr):
            if s==1:
                v, j = self.decode(idx)
                if checker[j]==1:
                    return False
                checker[j] += 1
            else:
                continue

        if all([c==1 for c in checker]):
            return True
        else:
            return False

    def check_edge_constraint(self, arr):
        edges = self.get_edges_from_arr(arr)
        if edges <= set(self.E):
            return True

    def check_constraints(self, arr):
        is_node_ok = self.check_node_constraint(arr)
        is_order_ok = self.check_order_constraint(arr)
        is_edge_ok = self.check_edge_constraint(arr)

        return is_node_ok, is_order_ok, is_edge_ok

    def solve_mysqa(self, trotter, T, gamma_schedule, save_points=[], save_dir='', preanneal=True):
        qdict = arr_to_Qdict(self.whole_qubo)
        solver = SQASolver()
        h, J, offset = qubo_to_ising(qdict)
        # dwave の表式とh,Jの符号が逆なので変換する
        for k, v in h.items():
            h[k] = -v
        for k, v in J.items():
            J[k] = -v
        solver.solve_ising(h, J,
                            trotter=trotter,
                            T=T,
                            gamma_schedule=gamma_schedule,
                            save_points=save_points,
                            save_dir=save_dir,
                            preanneal=preanneal)

    def save(self, path_dump):
        name = f"{self.__class__.__name__}.pickle"
        p = os.path.join(path_dump, name)
        with open(p, 'wb') as fp:
            pickle.dump(self, fp)

class TSP(HCP) :
    """Travel Salesman Problem
    """
    def __init__(self, Vset: set, Wdict: dict):
        """
            Args:
                Wdict (set): dictionary of edge-weight. e.g. { ('a', 'b'): 2, ('b', 'c'): 1.5, ...}
        """
        Eset = set(Wdict.keys())
        super().__init__(Vset, Eset)
        self.Wdict = Wdict
        self.W = dict()
        for e, w in Wdict.items():
            u = self.listedVset.index(e[0])
            v = self.listedVset.index(e[1])
            self.W[(u, v)] = w
        if not all( [e in self.E for e in self.W.keys()] ) : # check mappping for sure
            raise ValueError(f'mapping failed self.E {self.E} not matched with self.W.keys {list(self.W.keys())}')


    def set_weight_qubo(self):
        qubo = np.zeros(self.qubo_shape)
        for e, w in self.W.items():
            if e[0] == 0 : # start from 0
                qubo[self.encode(e[1], 1), self.encode(e[1], 1)] += w
            elif e[1] == 0 : # return to 0
                qubo[self.encode(e[0], self.N-1), self.encode(e[0], self.N-1)] += w
            else:
                for j in range(1, self.N-1):
                    qubo[self.encode(e[0], j), self.encode(e[1], j+1)] += w

        self.weight_qubo = qubo

    def set_whole_qubo(self, L, A, B, C):
        """set the weight qubo and the three constraints. overrade of HCP.set_whole_qubo method
        """
        self.L = L
        self.A = A
        self.B = B
        self.C = C

        self.set_weight_qubo()
        self.set_node_constraint()
        self.set_order_constraint()
        self.set_edge_constraint()

        self.whole_qubo = L*self.weight_qubo + A*self.node_constraint + B*self.order_constraint + C*self.edge_constraint

class Classification:

    N = 2

    def classify_state(state, hcp):
        state = spin_to_q(state)
        if all(hcp.check_constraints(state)):
            return 1
        else:
            return 0

