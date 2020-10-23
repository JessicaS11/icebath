import shapely.geometry
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
