# SOM, GEOSOM, OSM

This section is devolved to the many trials and tribulations to implement a simple Self-Organizing Map (SOM) and then a Geo-SOM¹.

A folder is dedicated to the tries made with R, which were deemed too slow to continue (limits of being a geographer and not a programmer).

Another folder contains the trials made with python and the final package obtained (geosom_package) which can be used to perform geosom on tabular (geo)data, visualise geosom results and characteristics, cluster data according to the geosom results and write them to file. Some trials with example data are included.

Some feature selection tries (PFA, Gaussian mixture models, FANNY) are included in a third folder.

Finally, the GeoSOM package has been used on building data from OpenStreetMap and _swisstopo_. The whole procedure of how to extract the buildings, retrieve their heights and compute all the indicators related to their shape² is described in the corresponding file. The geosom analysis is in the geoOSM file and contains the geosom analysis for both OSM and swisstopo data. For the OSM data, several thresholds for _k_ (size of neighbouring neurons to look in) and parameters (nb of iterations, $\alpha$, radius) are tested. Some tests are also performed while modifying neurons which do not have data points attached to them after the iterations. At the end a _K-Means_ clustering is performed before exporting the results for further analysis or display in another software. An example of some results is available as a .png file.

Several tries have been made on selected feature according to gaussian mixture models. The final selection originated either from handpicked features or from the full set. Some of the trials are included as well as results examples.

### Potentially useful excerpts
- GeoSOM package written in python (analysis, visualisation, export)
- Multidimensional array and parallel processing in R (and ineffective GeoSOM)
- Feature selection techniques with R
- Extracting information from OSM with R
- Computing more or less advanced shape attributes with postgis from R
- DB i/o from R
- GeoSOM analyses and several examples


¹ Bação, F., Lobo, V., & Painho, M. (2005). The self-organizing map, the Geo-SOM, and relevant variants for geosciences. _Computers & Geosciences_, _31_(2), 155-163.
² Schirmer, P. M., & Axhausen, K. W. (2016). A multiscale classification of urban morphology. _Journal of Transport and Land Use_, _9_(1), 101-130.
