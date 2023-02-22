import numpy as np

def arr_to_Qdict(arr):
    """turn back buttom trianguler part into the upper part
    """
    nhor = len(arr)
    nver = len(arr[0])

    dct = {}
    for u in range(nhor):
        for v in range(u, nver):
            if u==v:
                dct[(u, u)] = arr[u, u]
            else:
                dct[(u, v)] = arr[u, v] + arr[v, u]

    return dct

def spin_to_q(spin_config):
    """convert spin array ( [+1,-1]^n ) into q array ( [1, 0]^n )"""
    return np.array([1 if spin_config[i]==1 else 0 for i in range(len(spin_config))])
