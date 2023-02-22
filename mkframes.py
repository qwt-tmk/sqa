import os
import json
import pickle
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from problems.tsp import Classification
from utils.vis import plot_state2D

def load_npz(dir):
    npz = np.load(os.path.join(dir, 'states.npz'))
    return npz

def load_gammas(dir):
    with open(os.path.join(dir, 'gammas.json'), 'r') as fp:
        gammas = json.load(fp)
    return gammas

def load_tsp(dir):
    with open(os.path.join(dir, 'TSP.pickle'), 'rb') as fp:
        tsp = pickle.load(fp)
    return tsp

def get_newest_res(path_to_results):
    return os.path.join(path_to_results, sorted(os.listdir(path_to_results))[-1])

def make_frames(hcp, npz, gammas, save_dir):
    """
    Args:
        tsp (problems.tsp.HCP) : the used problem. TSP is acceptable off course.
        npz (np.npz) : the process of the system during solving the problem
        gammas (dict) : key-value is mcs-gamma
    """
    print(f"===== making frames to {save_dir}=====")
    Nframe = len(npz)
    for nframe, mcs in enumerate(npz.keys()):
        print(f"\r{nframe} / {Nframe}", end="")
        fig, ax = plt.subplots()
        state2D = npz[mcs]
        classify_state = partial(Classification.classify_state, hcp=hcp)
        N_classify = Classification.N
        c = plot_state2D(state2D, classify_state, N_classify, ax)
        cbar = fig.colorbar(c, ticks=list(range(2*N_classify)))
        cbar.ax.set_yticklabels(['down', 'up', 'sat down', 'sat up'])
        ax.set_xlabel("spin index")
        ax.set_ylabel("tortter index")
        ax.set_title(f"MCS:{mcs}\n gamma:{gammas[mcs]:.5f}")
        fig.savefig(os.path.join(save_dir,f"frame_{nframe:06d}.png"), bbox_inches="tight")
        plt.close(fig)
    print("\n====== done ======")


if __name__=="__main__":
    results_dir = "./tmp"
    newest = get_newest_res(results_dir)
    tsp = load_tsp(newest)
    npz = load_npz(newest)
    gammas = load_gammas(newest)
    save_dir = os.path.join(newest, 'frames')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    make_frames(tsp, npz, gammas, save_dir)
