
import os
import sys
import traceback
import json

import numpy as np

from sqa.system import System

class SQASolver:

    def __init__(self):
        pass

    def solve_ising(self, h, J, trotter, T, gamma_schedule, save_points=[], save_dir='', preanneal=True):
        """
        Args:
            h (dict) : linear biases as dict.
            J (dict) : quadratic coupling strengthes as dict.
            T (float) : temparature of the system.
            gamma_schedule(list of tuples) : list of (MCS, gamma) tuples.
            save_points (list) : list of the monte colro steps at which you want to save the system configuration.
            save_dir (str) : where to save configurations.
            preanneal (bool) : whether you execute preannealing by SA on the classical hamiltonian. if False totally random through spin axis and trotter axis.
        Return:
            (none) : this method does not return anything.

        """

        self.rng = np.random.default_rng()
        self.system = System(h, J, trotter, T, gamma_schedule[0][1])
        self.save_dir = save_dir
        self.state_memory = {}
        self.gamma_memory = {}

        if preanneal:
            self.preanneal()

        mcs_total = 0
        save_points = {int(sp) for sp in save_points}

        try:
            print("=====progress=====")
            coordinates = [(t, i) for t in range(trotter) for i in range(self.system.N)]
            for steps, gamma in gamma_schedule:
                mcs_total += steps
                print(f'\rmcs_total: {mcs_total}, gamma: {gamma}', end='')

                self.system.gamma = gamma
                for mcs in range(steps) :
                    self.rng.shuffle(coordinates)
                    for at in coordinates:
                            self.update(at)

                # save state and gamma to memory
                try:
                    save_point = min(save_points)
                except :
                    continue
                if mcs_total >= save_point:
                    self.state_memory[str(mcs_total)] = self.system.state.copy()
                    self.gamma_memory[str(mcs_total)] = self.system.gamma
                    save_points.discard(save_point)
            print("\n=====done=====")

        except Exception as e:
            print("\nGet a exception during monte calro")
            print(e)
            tb = sys.exc_info()[2]
            traceback.print_tb(tb)

        finally:
            # save memory to a npz file
            if len(self.state_memory) > 0:
                saved_path = self.save_memory()
                print(f"saved to {saved_path}")

    def save_memory(self):
        saved_path = os.path.join(self.save_dir, 'states')
        np.savez_compressed(saved_path, **self.state_memory)
        with open(os.path.join(self.save_dir, 'gammas.json'), 'w') as fp:
            json.dump(self.gamma_memory, fp)
        return saved_path

    def preanneal(self):
        """preanneal and return local configration. Boltzmann constant is assumed 1.
        """
        Teff = self.system.T * self.system.trotter
        temparatures = np.arange(Teff, Teff+2.05, 0.05)[::-1] # 有効温度3からパラメータで指定した有効温度までSA
        rng = np.random.default_rng()
        indices = np.arange(self.system.N)
        now = rng.choice([1, -1], size=self.system.N)

        for t in temparatures:
            beta = 1 / t
            rng.shuffle(indices)
            for mcs in range(100) :
                for i in indices:
                    # 1 MCS
                    candidate = now.copy()
                    candidate[i] = -1 * candidate[i]
                    deltaE = self.system.classical_E(candidate) - self.system.classical_E(now)
                    if deltaE <= 0 :
                        now = candidate.copy()
                    else:
                        r = rng.random()
                        if r <= np.exp( -1 * beta * deltaE ):
                            now = candidate.copy()

        self.system.state = np.array( [now.copy() for i in range(self.system.trotter)] )

    def update(self, at):
        delta = self.system.delta_effective_E(at=at)
        if delta <= 0 :
            self.system.flip_at(at)
        else :
            Teff = self.system.trotter*self.system.T
            if self.rng.random() < np.exp( -1 * delta / Teff) :
                self.system.flip_at(at)
