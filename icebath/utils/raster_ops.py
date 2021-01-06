import shapely.geometry
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from skimage import feature, morphology, filters, segmentation
from scipy import ndimage


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

    elev = elev#np.nan_to_num(elev, nan=0)
    # A particular challenge of using contour and handling nans in this application is that:
    # if nans are 0s, no data regions are outlined. It's more accurate to leave the nans
    # (which are handled by contour) and get fewer polygons. However, then the challenge arises
    # that in many DEMs, sea level > threshold by a lot, so we don't pick up icebergs because there's
    # not a large enough elevation difference in the pixels that ARE there for contour to pick up.
    X,Y = np.meshgrid(x,y)
    fig = plt.figure()
    cs = plt.contour(X, Y, elev, 'k-', levels=[threshold])
    # Can use collections[0] so long as you only have one threshold value. Otherwise you have to iterate through the collections.
    # print(cs.collections[0].get_alpha())
    # cs.collections.get_alpha()
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
    # 

    for contour_path in p: 
        # try to get the alpha value or otherwise determine if the polygon traces a nan region
        # print(contour_path.get_alpha)

        # The first polygon in the path is the main one, the following ones are "holes"
        for ncp,cp in enumerate(contour_path.to_polygons(closed_only=False)):
            if ncp>0:
                print('multiple contour paths detected - holes will be removed later')
           
            # check to make sure the polygon is a closed loop
            if cp[-1,0] == cp[0,0] and cp[-1,1] == cp[0,1]:
                polygons.append(cp)


    print(len(polygons))
    plt.close(fig)

    return polygons


def labeled_from_edges(elev,sigma,resolution,min_area, flipax=[]):
    '''
    Create an edge map, refine it, and use it to get a list of polygon vertices

    This function does not consider or deal with projections.

    Parameters
    ----------

    '''

    # Compute the Canny filter
    edges = feature.canny(elev, sigma=sigma)

    filled_edges = ndimage.binary_fill_holes(edges)

    # remove small objects and polyganize
    # if we assume a minimum area of 4000m2, then we need to divide that by the spatial 
    # resolution (2x2=4m2) to get the min size
    # Note: trying to polygonize the edge map directly is computationally intensive
    labeled,_ = ndimage.label(morphology.remove_small_objects(
        filled_edges, min_size=min_area/(resolution**2), connectivity=1))#[0] # 2nd output is num of objects
    # Note: can do the remove small objects in place with `in_place=False`
    # print(count)

    # flip the array, if needed
    labeled = np.flip(labeled, axis=flipax)
    
    return labeled

def labeled_from_segmentation(elev, markers, resolution, min_area, flipax=[]):
    """
    Apply some image filters, provide seed points using value extremes, and
    segment the image using a watershed analysis of the elevation map.

    This function does not consider or deal with projections.

    Parameters
    ----------

    markers: list
            ordered list with lower and upper values for seeding the segmentation
    """

    # Compute the "elevation map"
    elev_map = filters.sobel(elev, mask=~np.isnan(elev))

    # create seed markers and mask them
    marker_arr = np.zeros_like(elev)
    marker_arr[elev < markers[0]] = 1
    marker_arr[elev > markers[1]] = 2
    # marker_arr = np.ma.array(marker_arr, mask=np.isnan(elev))

    # create a watershed segmentation
    segmented = segmentation.watershed(elev_map, markers=marker_arr, mask=~np.isnan(elev))
    segmented[segmented<=1] = 0
    segmented[segmented==2] = 1

    # fill in the segmentation and label the features
    filled_seg = ndimage.binary_fill_holes(segmented)
    no_sm_bergs = morphology.remove_small_objects(filled_seg, min_size=min_area/(resolution**2.0), connectivity=1)
    lg_bergs = morphology.remove_small_objects(filled_seg, min_size=1000000/(resolution**2.0), connectivity=1)
    filled_bergs = np.logical_xor(no_sm_bergs, lg_bergs)

    labeled=ndimage.label(filled_bergs)[0]

    # flip the array, if needed
    labeled = np.flip(labeled, axis=flipax)
    
    return labeled

def test_feature(feat):
    # print(feat)
    bord_px = np.count_nonzero(feat>-888)
    nan_px = np.count_nonzero(np.isnan(feat)) # (feat==-999)

    print('feature count values')
    print(bord_px)
    print(nan_px)
    print(np.size(feat))
    # print(np.unique(feat))
    
    if bord_px==0 or np.float(nan_px)/bord_px >= 0.5:
        print('too many nan')
        return 0
    else:
        print('good berg')
        return 1

def border_filtering(feature_arr, input_arr, flipax=[]):
    """
    Dilate the potential iceberg features and remove those that have
    more than 50% of their border pixels on a no-data boundary.

    This function is modified from JessicaS11/iceberg_delin/icebergdelineation.py
    
    Parameters
    ----------

    feature_arr: ndarray, int
                Array of labeled features (e.g. from ndimage.label) to check the borders of
    input_arr: ndarray, float or int
                Array of original or classed data, same size and shape as feature_arr,
                that will be used to get border pixel values 
    """

    # make original features a dummy value in the dataset so they're easy to not count
    # print(feature_arr)
    # print(np.shape(feature_arr))
    # print(feature_arr[3500:3550][3450:3550])
    # print('look for nans in input_arr')
    # print(np.any(np.isnan(input_arr)))
    # print(np.any(np.isnan(feature_arr)))
    # print(np.unique(input_arr))
    input_arr[feature_arr>0] = -888 # features masked in original data
    # input_arr[np.isnan(input_arr)] = -999 # nodata values masked in original data

    # print(np.any(np.isnan(input_arr)))
    # dilate the features using a 1 pixel plus structuring element
    dilated_feats = ndimage.grey_dilation(feature_arr, 
                                            structure=ndimage.generate_binary_structure(2,1)).astype(feature_arr.dtype)
    # print(dilated_feats[3500:3550][3450:3550])
    # iterate through the features and determine if each one should be included (1) or not (0)
    num_feats_rng = np.arange(1, np.nanmax(dilated_feats)+1)
    keep_feat_idx = ndimage.labeled_comprehension(input_arr, dilated_feats, num_feats_rng, test_feature, int, -1)

    # generate a keep_feature_array binary "mask" that matches the original feature_arr shape
    sort_idx = np.argsort(num_feats_rng)
    keep_feat_arr = keep_feat_idx[sort_idx][np.searchsorted(num_feats_rng, dilated_feats, sorter=sort_idx)]

    # apply the mask to remove unwanted features
    labeled = keep_feat_arr*feature_arr

    # flip the array, if needed
    labeled = np.flip(labeled, axis=flipax)

    return labeled