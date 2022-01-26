import numpy as np
import numpy.typing as npt

def cuboidVolume(bb : npt.NDArray) -> npt.NDArray:

    dis = (bb[[0, 1, 4, 5],:] - bb[[2, 3, 2, 3],:])**2

    volume = (bb[9,:]-bb[8,:]) * np.sqrt( (dis[0,:]+dis[1,:]) * (dis[2,:]+dis[3,:]) )

    return volume