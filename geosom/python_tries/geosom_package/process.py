#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
import csv

class Data:
    def __init__(self, file):
        """Reads the data from the TSV file"""
        self.data = {}
        with open(file) as f:
            self.header = f.readline().strip().split("\t")
            for l in f.readlines():
                v = l.strip().split("\t")
                self.data[v[0]] = list(map(float, v[1:]))

    def values(self):
        return self.data.values()

    @property
    def minval(self):
        """Computes the minimum value for each dimension"""
        return np.min(list(self.data.values()), axis=0)

    @property
    def maxval(self):
        """Computes the minimum value for each dimension"""
        return np.max(list(self.data.values()), axis=0)

    @property
    def ncols(self):
        """Returns the number of data columns"""
        return len(list(self.data.values())[0])

def georandinit(ndims, data):
    """
    Creates the grid for the code vectors (x,y,n, ..., m) and makes a random
    initialization (using a uniform distribution over the data domain) for the
    non-geo data.
    """
    # Determine the dimensions of the code vector array
    dims = list(ndims) + [data.ncols]

    # Find the sequence of x and y coords according to the dimension
    xCoords = np.linspace(data.minval[0], data.maxval[0], ndims[0])
    yCoords = np.linspace(data.minval[1], data.maxval[1], ndims[1])

    # Generate random values between 0 and 1
    cod = np.random.random_sample(dims)
    # Get the minimum and maximum values for every data column
    vmin, vmax = np.array(data.minval[2:]), np.array(data.maxval[2:])
    # Regular assigment of the x and y coordinates
    for col in range(ndims[1]):
        for li in range(ndims[0]):
            cod[li,col, 0:2] = (xCoords[li], yCoords[col])
    # Adjust the random values to be within the data domain for each column
    #final = cod * (vmax - vmin) + vmin
    cod[:,:,2:] = cod[:,:,2:] * (vmax - vmin) + vmin

    return cod

def gaussian_for_radius(radius):
    # Compute the squared distance for the radius matrix
    a = np.ones(shape=(radius*2+1, radius*2+1))
    for i in range(radius*2+1):
        for j in range(radius*2+1):
            a[i,j] = (i-radius)**2 + (j-radius)**2
    # Compute the Gaussian function
    return np.exp((-1 * a) / (2. * radius * radius))

def find_bmu(cod, d, geo = False):
    """
    Return the best (geo)matching unit index given a map and a code vector.
    Geo parameter switches between spatial and attribute distance.
    """
    if geo:
        dist = np.sum(np.power(cod[:,:,:2] - d[:2], 2), axis = 2)
    else:
        dist = np.sum(np.power(cod[:,:,2:] - d[2:], 2), axis = 2)

    return np.unravel_index(np.argmin(dist, axis=None), dist.shape)

def find_nb_bmu(cod, d, gbmu_idx, r):
    """
    Return the best matching unit index within a radius r of a geographic BMU
    given a map, code vector and the geoBMU index.
    """
    # Extract the shape and convert idx
    som_dim = cod.shape[0:2]
    gbmu_idx = np.array(gbmu_idx)

    # Deduct the coordinates of the (reduced) map where to find the bmu
    ccmin = max(0, gbmu_idx[0] - r)
    ccmax = min(som_dim[0], gbmu_idx[0] + r + 1)
    clmin = max(0, gbmu_idx[1] - r)
    clmax = min(som_dim[1], gbmu_idx[1] + r + 1)
    sm_cod = cod[ccmin:ccmax, clmin:clmax]

    gbmu_idx_sm = np.array(find_bmu(sm_cod, d))
    bmu_idx = tuple(gbmu_idx_sm + np.array((ccmin, clmin)))

    return(bmu_idx)

def adapt_cod_around_bmu_geo(cod, d, bmu_idx, radius, gauss, alpha, updateOnlyNonGeo = True):
    """
    Update the code vector according to the given vector and bmu index.
    Possible to also update the coordinates if updateOnlyNonGeo is False
    """
    # Extract the shape and convert idx
    som_dim = cod.shape[0:2]
    bmu_idx = np.array(bmu_idx)

    # Deduct the coordinates of the cod to adapt
    ccmin = max(0, bmu_idx[0] - radius)
    ccmax = min(som_dim[0], bmu_idx[0] + radius + 1)
    clmin = max(0, bmu_idx[1] - radius)
    clmax = min(som_dim[1], bmu_idx[1] + radius + 1)
    sm_cod = cod[ccmin:ccmax, clmin:clmax]
    #cmin = np.maximum(bmu_idx - radius, [0,0])
    #cmax = np.minimum(bmu_idx + radius, som_dim)

    # Check if cod is smaller than the gaussian
    if sm_cod.shape[0:2] != gauss.shape:
        # Get the indices for the part of the Gaussian function matrix
        # to consider
        cmin = np.array((ccmin, clmin))
        cmax = np.array((ccmax, clmax))
        gmin = radius - (bmu_idx - cmin)
        gmax = radius + (cmax - bmu_idx)
        sm_gauss = gauss[gmin[1]:gmax[1], gmin[0]:gmax[0]]
        # Reshape the adapted gaussian to allow element-wise multiplication
        sm_gauss = np.reshape(np.transpose(sm_gauss), (sm_cod.shape[0], sm_cod.shape[1], 1))
    else:
        # Reshape the gaussian to allow element-wise multiplication
        sm_gauss = np.reshape(gauss, (sm_cod.shape[0], sm_cod.shape[1], 1))

    # Check if coordinates must be updated
    if updateOnlyNonGeo:
        cod[ccmin:ccmax, clmin:clmax, 2:] = cod[ccmin:ccmax, clmin:clmax, 2:] + sm_gauss * alpha * (d[2:] - cod[ccmin:ccmax, clmin:clmax, 2:])
    else:
        # W(t+1) = W(t) + gauss * alpha * (data - code vector)
        cod[ccmin:ccmax, clmin:clmax] = cod[ccmin:ccmax, clmin:clmax] + sm_gauss * alpha * (d - cod[ccmin:ccmax, clmin:clmax])

def vgeosom(data, cod, rlen=1000, alpha=0.5, radius=10, k = 1, keepCoordinates = True):
    """
    Trains the SOM defined in `cod` with the data from `data`,
    with a run length `rlen`. Learning rate `alpha` and
    search radius `radius` are decreased linearly to 0 respectively 1
    during learning. k is the geotolerance and keepCoordinates indicates
    whether the geographic features of the cod shall be updated or not.
    Topology type is squares, and neighborhood function is Gaussian.

    TODO ONCE THE GEOBMU IS FOUND ADD THE INDEX TO THE DATAPOINTS TO AVOID
    COMPUTING THE SPATIAL DISTANCE AT EACH ITERATION !!!

    """
    # Extract the data array and convert it to a numpy array
    data_array = np.array(list(data.values()))
    # Iterate
    for i in range(rlen):
        cur_alpha = alpha * (1. - (i*1. / rlen))
        cur_radius = int(np.round(1 + (radius-1) * (1. - (i*1. / rlen))))
        # Shuffle the data array before each iteration
        np.random.shuffle(data_array)
        # Precompute the Gaussian function for the current radius
        gaussian = gaussian_for_radius(cur_radius)
        # For each row in the data set compute the BMU
        for d in data_array:
            # Find geo bmu
            geo_bmu = find_bmu(cod, d, geo = True)
            # Find BMU in geo bmu neighbours
            bmu_idx = find_nb_bmu(cod, d, geo_bmu, k)
            #
            adapt_cod_around_bmu_geo(cod, d, bmu_idx, cur_radius, gaussian, cur_alpha, updateOnlyNonGeo = keepCoordinates)

def kmns_on_cod(cod, k_nbrs, geo = False):
    """
    Effectue un kmean sur le cod
    """
    if not geo:
        cod = cod[:,:,2:]
    flat_cod = cod.reshape((cod.shape[0] * cod.shape[1], cod.shape[2]))
    km = KMeans(n_clusters=k_nbrs).fit(flat_cod)

    return km

def cah_on_cod(cod, nb_clus = 1, geo = False):
    """"
    Commentaire
    """
    if not geo:
        cod = cod[:,:,2:]
    flat_cod = cod.reshape((cod.shape[0] * cod.shape[1], cod.shape[2]))
    l_mat = linkage(flat_cod, 'ward')

    plt.figure(figsize=(100, 100))
    plt.title('HCA Dendrogram')
    plt.ylabel('distance')
    dendrogram(l_mat, labels=None)
    plt.show()

    return cut_tree(l_mat, n_clusters=nb_clus)

def attribution_array(data, cod, labs_km_cod, geo = False, k = 1):
    """
    Retourne une liste qui correspond au groupe attribué pour chacune des
    observations contenues dans les données initiales
    Doit être effectué sur le cod calculé, avec les labels du kmeans obtenus
    avec kms_rslt
    """
    data_array = np.array(list(data.values()))
    group = [None] * len(data_array)

    for i in range(0, len(data_array)):
        bmu = find_bmu(cod=cod, d=data_array[i], geo = geo)
        if geo:
            bmu = find_nb_bmu(cod=cod, d=data_array[i], gbmu_idx=bmu, r = k)

        group[i] = labs_km_cod[bmu]

    return group

def export_cat_csv(cat_list, filename):
    """
    Exporte la liste des groupes en format csv
    """
    with open(filename, "w", newline="\n") as fout:
        writer = csv.writer(fout, delimiter = ',')
        for i in cat_list:
            writer.writerows([[i]])

def modify_cod_without_hit(cod, data, geo = False, k = 1, defaultValue = -9999):
    """
    Change les valeurs du cod où il n'y a pas de hit.
    Permet d'atténuer l'influence de ces valeurs sur la classification
    """
    data_array = np.array(list(data.values()))

    hits = np.zeros(cod.shape[:2], dtype=int)

    for d in data_array:
        bmu = find_bmu(cod, d, geo = geo)
        if geo:
            bmu = find_nb_bmu(cod, d, bmu, k)
        hits[bmu] = hits[bmu] + 1

    adjusted_cod = np.where(np.reshape([hits>0][0], (cod.shape[0], cod.shape[1], 1)), cod, np.ones(cod.shape, dtype = int) * defaultValue)

    return adjusted_cod

def modify_cod_with_kmrslt(cod, kms_labs):
    """
    Ajoute le résultat du kmeans aux neurones en vue d'un export
    """
    km_rs = kms_labs.reshape(cod.shape[0], cod.shape[1], 1)
    modified_cod = np.append(cod, km_rs, axis = 2)

    return modified_cod

def export_cod_to_csv(cod, filename):
    """
    Exporte les valeurs du cod en fichier .csv
    """
    cod_exp = cod.reshape(cod.shape[0] * cod.shape[1], cod.shape[2])
    np.savetxt(filename, cod_exp, delimiter=',')






















# fd
