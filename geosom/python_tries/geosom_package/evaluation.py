import numpy as np
from geosom_pck.process import find_bmu

def qerror(data, cod):
    """
    Compute the mean geo quantization error and the mean data quantization error
    """
    data_array = np.array(list(data.values()))
    q_geo = np.zeros(data_array.shape[0])
    q_dat = np.zeros(data_array.shape[0])

    for i in range(data_array.shape[0]):
        d = data_array[i]
        g_bmu = find_bmu(cod, d, geo = True)
        d_bmu = find_bmu(cod, d)

        g_dist = np.sum(np.power(cod[g_bmu[0], g_bmu[1], :2] - d[:2], 2))
        d_dist = np.sum(np.power(cod[d_bmu[0], d_bmu[1], 2:] - d[2:], 2))

        q_geo[i] = g_dist
        q_dat[i] = d_dist

    m_qgeo = np.mean(q_geo)
    m_qdat = np.mean(q_dat)

    return [m_qgeo, m_qdat]
