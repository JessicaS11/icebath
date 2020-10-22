import shapely.geometry
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def poly_from_thresh(x,y,elev,threshold, mask=None):
    '''
    Threshold a raster dataset and return all closed polygons.

    Function based on Mike Wood's iceberg_outlines function

    If a mask path is provided, it must be in the same CRS as the data to be thresholded. This function
    does not consider or deal with projections.

    Parameters
    ----------

    mask : array of shape N,2
    x,y coordinates defining a mask path in the same coordinate system as the x, y data.
    '''

    # remove nan values from elev array
    elev = np.nan_to_num(elev, nan=0)

    if mask is not None:
        mask = mpl.Path(mask)

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


    print(len(p))
    # for pi in range(len(p)):
    #     polygon = p[pi].vertices
    #     #check to make sure the polygon is a closed loop
    #     if polygon[-1,0] == polygon[0,0] and polygon[-1,1] == polygon[0,1]:
    #         polygons.append(polygon)

    for contour_path in p: 
        # The first polygon in the path is the main one, the following ones are "holes"
        for ncp,cp in enumerate(contour_path.to_polygons(closed_only=False)):
            # print(type(cp))
            # x = cp[:,0]
            # y = cp[:,1]
            if ncp>0:
                print('multiple contour paths detected - holes will be removed later')
           
            # check to make sure the polygon is a closed loop
            if cp[-1,0] == cp[0,0] and cp[-1,1] == cp[0,1]:
                # apply mask
                if mask is not None:
                    if cp.intersects_path(mask, filled=True):
                        pass
                    else: polygons.append(cp)          
                else:
                    polygons.append(cp)

            

        
            
            
            # # shapely implicitly closes any linear rings that don't have matching first/last vertices
            # new_shape = shapely.geometry.Polygon([(i[0], i[1]) for i in zip(x,y)])
            # # if new_shape.is_valid:
            # #     poly = new_shape
            # # else:
            # #     print(new_shape)
            # #     poly = new_shape.buffer(0)
            # #     print(poly)
            
            # if ncp == 0:
            #     poly = new_shape
            #     # print('had multiple contour paths')
            # else:
            #     # Remove the holes if there are any
            #     poly = new_shape.difference(new_shape)
            #     # print('had to remove a hole')
            #     # Can also be left out if you want to include all rings
            # # print(len(poly.exterior.coords))

            # polygons.append(list(poly.exterior.coords))


    print(len(polygons))
    # print(type(polygons))

    plt.close(fig)

    return polygons
