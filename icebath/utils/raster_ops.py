import matplotlib.pyplot as plt
import numpy as np

def poly_from_thresh(x,y,elev,threshold):
    '''
    Threshold a raster dataset and return all closed polygons.

    Function based on Mike Wood's iceberg_outlines function

    Parameters
    ----------
    '''

    
    X,Y = np.meshgrid(x,y)
    fig = plt.figure()
    cs = plt.contour(X, Y, elev, 'k-', levels=[threshold])
    p = cs.collections[0].get_paths()
    polygons = []
    for pi in range(len(p)):
        polygon = p[pi].vertices
        #check to make sure the polygon is a closed loop
        if polygon[-1,0] == polygon[0,0] and polygon[-1,1] == polygon[0,1]:
            polygons.append(polygon)
    plt.close(fig)

    return polygons
