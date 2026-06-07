import numpy as np

def L2(SG_syn, SG_obs) -> tuple[np.ndarray, np.ndarray]:

    adj_STFS = np.flip((SG_syn.gather) - (SG_obs.gather), axis=0)
    misfits = np.sum(adj_STFS**2, axis=0)

    return adj_STFS, misfits
