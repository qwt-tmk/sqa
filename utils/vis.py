
from colorsys import rgb_to_hsv, hsv_to_rgb
from typing import Callable

import numpy as np
from matplotlib.colors import ListedColormap


def plot_state2D(state2D, classify_state:Callable[[np.ndarray], int], N_classification:int, ax):
    """plot 2D state(configuration) with colors of classification.
        Args:
            state2D(list or np.ndarray): 2D state
            classify_state (Callable) : a function classifying state as integer code. the callable should take 1D state and return an integer from 0 to N_classification -1
            N_classification (int) : total number of classification
            ax : ax object
        Return:
            (matplotlib.collections.Collection) : result of matplotlib.pyplot.pcolor

        Examples:
            ::
                trotter = 40
                N = 10
                rng = np.random.default_rng()
                state2D = rng.choice([[1]*10, [-1]*10, rng.choice([1, -1], size=10)], size=trotter)

                def classify_state(state):
                    if all(state==1) or all(state==-1):
                        return 1
                    else:
                        return 0

                fig, ax = plt.subplots()
                c = plot_state2D(state2D, classify_state, 2, ax)
                cbar = fig.colorbar(c, ticks=[0, 1, 2, 3])
                cbar.ax.set_yticklabels(['down', 'up', 'ferro down', 'ferro up'])
                ax.set_xlabel("spin index")
                ax.set_ylabel("tortter index")
                ax.set_title("sk model")
                plt.show()
    """

    def convert_to_colorcode(state2D):
        colorcode2D = []
        for state in state2D:
            m = classify_state(state)
            colorcode = [ int(2*m+(spin+1)/2) for spin in state ]
            colorcode2D.append(colorcode)
        return colorcode2D

    def get_spin_palette(N_hue):
        rgb_list = [(1,0,0), (0,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,0)]
        if N_hue > len(rgb_list)+1:
            raise ValueError(f'N_hue must be less than or equal to {len(rgb_list)}')
        cp = []
        grey_up = hsv_to_rgb(0, 0, 0.6)
        grey_down = hsv_to_rgb(0, 0, 0.4)
        cp.extend([grey_up, grey_down])
        rgb_list = rgb_list[:N_hue-1]
        for rgb in rgb_list:
            hls = rgb_to_hsv(*rgb)
            hls_up = (hls[0], 0.3, 0.8)
            hls_down = (hls[0], 0.7, 0.8)
            cp.extend([hsv_to_rgb(*hls_up), hsv_to_rgb(*hls_down)])
        return ListedColormap(cp, name='spin')

    def plot_colorcode(colorcode2D, ax):
        N_hue = 1 + N_classification // 2
        cp = get_spin_palette(N_hue)
        return ax.pcolor(colorcode2D, cmap=cp, vmin=0, vmax=2*N_hue-1)

    cc = convert_to_colorcode(state2D)
    return plot_colorcode(cc, ax)
