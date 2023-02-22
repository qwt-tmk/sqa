
import os

import numpy as np

class System :
    """H = - sum_i h_i s_i - sum_i!=j J_ij s_i s_j
    """
    def __init__(self, h, J, trotter, T, gamma):
        rng = np.random.default_rng()

        self.h = h
        self.J = J
        self.trotter = trotter
        self.T = T
        self.gamma = gamma

        self.variables = list(self.get_vars(self.h, self.J))
        self.N = len(self.variables)
        self.hvec = np.array( [ h.get(var, 0) for var in self.variables ] )
        self.Jmatrix = self.get_Jmatrix(J)
        self.state = rng.choice([1, -1], size=(self.trotter, self.N) )

    def get_vars(self, h, J):
        vars_in_h = { v for v in h.keys() }
        vars_in_J = { v for vv in J.keys() for v in vv }
        return vars_in_h | vars_in_J

    def get_Jmatrix(self, J):
        Jmatrix = np.zeros((self.N, self.N))
        for uv, w in self.J.items():
            iu = self.variables.index(uv[0])
            iv = self.variables.index(uv[1])
            if iu != iv :
                Jmatrix[iu, iv] = w
        return Jmatrix

    def classical_E(self, classical_state:np.ndarray):
        classical_state = np.array(classical_state)
        E = 0
        E -= np.dot(self.hvec, classical_state)
        E -= np.dot(classical_state, np.dot(self.Jmatrix, classical_state))
        return E

    def effective_E(self, effective_state:np.ndarray):
        effective_state = np.array(effective_state)
        if effective_state.shape != self.state.shape :
            raise ValueError(f"input state's shape {effective_state.shape} does not match with instance shape {self.state.shape}")
        ferro = - self.trotter*self.T*np.log(np.tanh(self.gamma/(self.trotter*self.T) )) / 2 # > 0
        E = 0
        ss = 0
        slice_indices = list(range(self.trotter))
        for slice_idx, slice_next in zip(slice_indices, slice_indices[1:]+[0] ):
            # classical axis
            E += self.classical_E(effective_state[slice_idx])

            # trotter axis
            ss += sum( [ sk*_sk for sk, _sk in zip(effective_state[slice_idx], effective_state[slice_next]) ] )

        E -= ferro*ss
        return E

    def delta_effective_E(self, at):
        """Eeff_after - Eeff_before on one-spin flip

        Args :
            at : (tuple or list) : coordinate of target spin. (trotter-index, classical-index)
        """
        t = at[0]
        i = at[1]
        delta = 0

        # local part
        vert = self.Jmatrix[i, :].copy() # 行列のi行目
        vert[i] = 0
        hori  = self.Jmatrix[:, i].copy() # 行列のi列目
        hori[i] = 0
        delta += sum([self.state[t,i]*self.state[t,j]*vert[j] for j in range(self.N)])
        delta += sum([self.state[t,i]*self.state[t,j]*hori[j] for j in range(self.N)])
        delta += self.hvec[i]*self.state[t,i]

        # trotter part
        K = - self.trotter*self.T*np.log(np.tanh(self.gamma/(self.trotter*self.T) )) / 2 # > 0
        t_plus = t+1 if t<self.trotter-1 else 0
        t_minus = t-1 if 0<t else self.trotter-1
        delta += K * (self.state[t_plus,i] + self.state[t_minus,i]) * self.state[t,i]

        return 2*delta

    def flip_at(self, at):
        self.state[at] = -1*self.state[at]
