import shapely.geometry
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyproj


def crs2crs(incrs, outcrs, x, y, z):
    """
    Given two coordinate reference systems and x, y, z points, convert between them
    """

    # info on projections: pyproj.crs.CRS.from_epsg(3855)
    tf = pyproj.Transformer.from_proj(incrs, outcrs)
    gx, gy, gz = tf.transform(x,y,z)

    return gx, gy, gz


"""
# Original attempt at crs2crs to apply geoid offset using the Proj +vgridshift flag via pyproj, which didn't work

import os
# check that grid file to be used is actually present (no warning will be given to the user otherwise)
grid = 'egm08_25.gtx'
assert os.path.exists(pyproj.datadir.get_data_dir()+'/'+grid), "Grid file needed"
# otherwise, download the grid file: http://download.osgeo.org/proj/vdatum/egm08_25/
# and put it in the right directory: `pyproj.datadir.get_data_dir()`

import pyproj
from pyproj.transformer import TransformerGroup

# Find grids by using the map at: https://cdn.proj.org/
ellipsoidal = 'epsg:3413'
geoidal = '+init=epsg:3413 +proj=vgridshift +grids=egm08_25.gtx'

# Using a TransformGroup --> results in no vertical offset (but a km scale horizontal one)
tfgroup=TransformerGroup(ellipsoidal, geoidal, always_xy=True)
print(tfgroup)
newx, newy, newz = tfgroup.transformers[0].transform(x,y,z)
"""


def poly_from_thresh(x,y,elev,threshold):
    '''
    Threshold a raster dataset and return all closed polygons.

    Function based on Mike Wood's iceberg_outlines function

    This function does not consider or deal with projections.

    Parameters
    ----------

    '''

    elev = np.nan_to_num(elev, nan=0)
    X,Y = np.meshgrid(x,y)
    fig = plt.figure()
    cs = plt.contour(X, Y, elev, 'k-', levels=[threshold])
    # Can use collections[0] so long as you only have one threshold value. Otherwise you have to iterate through the collections.
    p = cs.collections[0].get_paths()
    polygons = []
    
    # DevNotes: Mike's original code iterated through each path in p, got the vertices, and stored it only if it was a closed loop.
    # The matplotlib docs suggest avoiding accessing a path's vertices directly, and a great StackExchange post noted potential issues
    # with returning lines and turned the paths into shapely.geometry.Polygon objects to help troubleshoot this. I'm going to allow
    # multiple rings for now in this step, and make sure that I don't have holes later on when I'm already converting to Polygons in 
    # icebath.core.build_gdf
    # https://gis.stackexchange.com/questions/99917/converting-matplotlib-contour-objects-to-shapely-objects
    # for pi in range(len(p)):
    #     polygon = p[pi].vertices
    # It is also recommended to now use scikit-image for contour finding (skimage.measure.find_contours)

    for contour_path in p: 
        # The first polygon in the path is the main one, the following ones are "holes"
        for ncp,cp in enumerate(contour_path.to_polygons(closed_only=False)):
            if ncp>0:
                print('multiple contour paths detected - holes will be removed later')
           
            # check to make sure the polygon is a closed loop
            if cp[-1,0] == cp[0,0] and cp[-1,1] == cp[0,1]:
                polygons.append(cp)

    # print(len(polygons))
    # print(type(polygons))
    plt.close(fig)

    return polygons
