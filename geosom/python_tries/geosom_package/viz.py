import numpy as np
import matplotlib.pyplot as plt
import math
from geosom_pck.process import find_bmu, find_nb_bmu

def plot_property(cod, col_number = '', geo = False):
    """
    Plots properties of a given cod vector.
    If a property number is given, the property heatmap is returned.
    Else all properties are mapped in smaller subplots and if the cod contains
    geocoordinates, geo must be set to True.
    """
    # No number given
    if col_number == '':
        # Number of properties
        col_numbs = cod.shape[2]
        # Set the geo coordinates aside
        if geo == True:
            col_numbs = cod[:,:,2:].shape[2]
        # Organise plot in three columns
        ncol = 3
        nrow = col_numbs/3
        # Adjust the number of rows
        if nrow % 3 != 0:
            nrow = math.ceil(nrow)
        # Define the number of subplots (grid form)
        fig, axeslist = plt.subplots(ncols = ncol, nrows = nrow)
        # Adjust spaces
        fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)

        for col in range(col_numbs):
            axeslist.ravel()[col].imshow(np.transpose(cod[:, :, col]), interpolation = "None", cmap = "Greys")
            axeslist.ravel()[col].set_title("Property " + str(col))
            #axeslist.ravel()[col].colorbar()
            axeslist.ravel()[col].set_axis_off()
        for outbounds in range(col_numbs, ncol*nrow):
            fig.delaxes(axeslist.ravel()[outbounds])
    else:
        plt.imshow(np.transpose(cod[:, :, col_number]), interpolation = "None", cmap = "Greys")
        plt.colorbar()
        plt.suptitle("Property " + str(col_number))

def compute_Umatrix(cod, geo = False, vois = 'tour'):
    """
    False Umatrix
    """
    # Pour chaque cellule, déterminer les voisins
    lmax, cmax = cod.shape[0]-1, cod.shape[1]-1
    dists = []
    for x,y in np.ndindex(cod.shape[:2]):
        ref = cod[x,y, 2:]

        if(x == 0 and y == 0):
            nbrs = np.array([
                cod[x+1, y, 2:],
                cod[x, y+1, 2:]
            ])
        elif(x == lmax and y == cmax):
            nbrs = np.array([
                cod[x-1, y, 2:],
                cod[x, y-1, 2:]
            ])
        elif(x == 0 and y == cmax):
            nbrs = np.array([
                cod[x+1, y, 2:],
                cod[x, y-1, 2:]
            ])
        elif(y == 0 and x == lmax):
            nbrs = np.array([
                cod[x-1, y, 2:],
                cod[x, y+1, 2:]
            ])
        elif(x == 0):
            nbrs = np.array([
                cod[x+1, y, 2:],
                cod[x, y-1, 2:],
                cod[x, y+1, 2:]
            ])
        elif(x == lmax):
            nbrs = np.array([
                cod[x-1, y, 2:],
                cod[x, y-1, 2:],
                cod[x, y+1, 2:]
            ])
        elif(y == 0):
            nbrs = np.array([
                cod[x-1, y, 2:],
                cod[x+1, y, 2:],
                cod[x, y+1, 2:]
            ])
        elif(y == cmax):
            nbrs = np.array([
                cod[x-1, y, 2:],
                cod[x+1, y, 2:],
                cod[x, y-1, 2:]
            ])
        else:
            nbrs = np.array([
                cod[x-1, y, 2:],
                cod[x+1, y, 2:],
                cod[x, y-1, 2:],
                cod[x, y+1, 2:]
            ])

    # Faire la moyenne des distances euclidiennes à ces voisins
        dist_tot = np.sum(abs(nbrs - ref))
        dist_mean = dist_tot / nbrs.shape[0]
        dists.append(dist_mean)
    # Dessiner la U-matrix

    dist_mat = np.reshape(np.array(dists), (cod.shape[0], cod.shape[1]), order = 'F')
    plt.imshow(dist_mat, interpolation = "None", cmap = "Greys")
    plt.colorbar()
    plt.suptitle("U-Matrix")

    return np.transpose(dist_mat)

def correctUmat(cod, geo = True, plot_nodes = False):
    """
    Compute and plot the U-Matrix of the given cod
    If geo is true, the coordinates are used for the distance computation
    Original nodes can be plotted on the U-Matrix
    """
    if geo:
        cod = cod[:, :, 2:]

    xmax, ymax = cod.shape[0]-1, cod.shape[1]-1
    umat = np.zeros((cod.shape[0]*2 - 1, cod.shape[1]*2 - 1, 1))

    for x, y in np.ndindex(cod[:,:,0].shape[:2]):
        if x < xmax:
            dist = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y,:], 2)))
            umat[2*x+1, 2*y] = dist
        if y < ymax:
            dist = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y+1,:], 2)))
            umat[2*x, 2*y+1] = dist
        if x < xmax and y < ymax:
            dist1 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y+1,:], 2)))
            dist2 = math.sqrt(np.sum(np.power(cod[x+1,y,:] - cod[x,y+1,:], 2)))
            umat[2*x + 1, 2*y + 1] = (dist1 + dist2)/2

        if x > 0 and y > 0 and x < xmax and y < ymax:
            dist1 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y+1,:], 2)))
            dist2 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y,:], 2)))
            dist3 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y-1,:], 2)))
            dist4 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y-1,:], 2)))
            dist5 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y-1,:], 2)))
            dist6 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y,:], 2)))
            dist7 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y+1,:], 2)))
            dist8 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y+1,:], 2)))
            umat[2*x, 2*y] = (dist1+dist2+dist3+dist4+dist5+dist6+dist7+dist8)/8
        elif x > 0 and x < xmax and y == 0:
            dist1 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y,:], 2)))
            dist2 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y+1,:], 2)))
            dist3 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y+1,:], 2)))
            dist4 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y+1,:], 2)))
            dist5 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y,:], 2)))
            umat[2*x, 2*y] = (dist1+dist2+dist3+dist4+dist5)/5
        elif x == xmax and y > 0 and y < ymax:
            dist1 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y+1,:], 2)))
            dist2 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y-1,:], 2)))
            dist3 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y-1,:], 2)))
            dist4 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y,:], 2)))
            dist5 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y+1,:], 2)))
            umat[2*x, 2*y] = (dist1+dist2+dist3+dist4+dist5)/5
        elif x > 0 and x < xmax and y == ymax:
            dist1 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y-1,:], 2)))
            dist2 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y,:], 2)))
            dist3 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y,:], 2)))
            dist4 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y-1,:], 2)))
            dist5 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y-1,:], 2)))
            umat[2*x, 2*y] = (dist1+dist2+dist3+dist4+dist5)/5
        elif x == 0 and y > 0 and y < ymax:
            dist1 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y+1,:], 2)))
            dist2 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y,:], 2)))
            dist3 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y-1,:], 2)))
            dist4 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y-1,:], 2)))
            dist5 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y+1,:], 2)))
            umat[2*x, 2*y] = (dist1+dist2+dist3+dist4+dist5)/5
        elif x == 0 and y == 0:
            dist1 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y,:], 2)))
            dist2 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y+1,:], 2)))
            dist3 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y+1,:], 2)))
            umat[x,y] = (dist1+dist2+dist3)/3
        elif x == 0 and y == ymax:
            dist1 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y-1,:], 2)))
            dist2 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x+1,y,:], 2)))
            dist3 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y-1,:], 2)))
            umat[x, 2*y] = (dist1+dist2+dist3)/3
        elif x == xmax and y == 0:
            dist1 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y+1,:], 2)))
            dist2 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y+1,:], 2)))
            dist3 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y,:], 2)))
            umat[2*x, y] = (dist1+dist2+dist3)/3
        elif x == xmax and y == ymax:
            dist1 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y,:], 2)))
            dist2 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x-1,y-1,:], 2)))
            dist3 = math.sqrt(np.sum(np.power(cod[x,y,:] - cod[x,y-1,:], 2)))
            umat[2*x, 2*y] = (dist1+dist2+dist3)/3

    umat_viz = plt.imshow(np.transpose(umat[:,:,0]), interpolation = "None", cmap = "Greys")
    plt.colorbar()
    if plot_nodes:
        x_coord = np.arange(0, cod.shape[0]*2 - 1, 2)
        x_coord_np = np.repeat(x_coord, cod.shape[1])
        y_coord = np.arange(0, cod.shape[1]*2 - 1, 2)
        y_coord_np = np.ravel(np.repeat(np.reshape(y_coord, (1, y_coord.shape[0])), cod.shape[0], axis = 0))
        plt.scatter(x_coord_np, y_coord_np, marker = 's')
    plt.show()

    return umat

def hitmap(cod, data, geo = False, r = 1):
    """
    Compute and plot the hitmap of the data vectors on the SOM
    If geo is true, first the geoBMU is found and then, in the
    neighbourhood defined by k, the BMU is used for the "hit"
    """
    data_array = np.array(list(data.values()))

    hits = np.zeros(cod.shape[:2], dtype=int)

    for d in data_array:
        bmu = find_bmu(cod, d, geo = geo)
        if geo:
            bmu = find_nb_bmu(cod, d, bmu, r)
        hits[bmu] = hits[bmu] + 1

    plt.imshow(np.transpose(hits), interpolation = "None", cmap = "Greys")
    plt.colorbar()

    return np.transpose(hits)

def kms_rslt(km, cod):
    labs = km.labels_.reshape((cod.shape[0], cod.shape[1]))
    plt.imshow(np.transpose(labs), interpolation="None", cmap="Set1")
    plt.colorbar()

    return labs

def cah_rslt(cut_tree, cod, k):

    for i in range(0, cut_tree.shape[1]):
        labs = cut_tree[:, i].reshape((cod.shape[0], cod.shape[1]))
        plt.imshow(np.transpose(labs), interpolation="None", cmap="tab20")
        plt.colorbar()
        plt.title("CAH avec " + str(len(np.unique(cut_tree[:, i]))) + " clusters")
        plt.show()

    if k > cut_tree.shape[1]:
        raise ValueError('K supérieur à la dimension du résultat de la CAH')

    labs = cut_tree[:, k].reshape((cod.shape[0], cod.shape[1]))

    print("Résultat pour " + str(len(np.unique(cut_tree[:, k]))) + " clusters retourné")
    return labs
