import shapely.geometry
import matplotlib.pyplot as plt
import numpy as np

def poly_from_thresh(x,y,elev,threshold):
    '''
    Threshold a raster dataset and return all closed polygons.

    Function based on Mike Wood's iceberg_outlines function

    Parameters
    ----------
    '''

    # remove nan values from elev array
    elev = np.nan_to_num(elev, nan=0)

    X,Y = np.meshgrid(x,y)
    fig = plt.figure()
    cs = plt.contour(X, Y, elev, 'k-', levels=[threshold])
    p = cs.collections[0].get_paths()
    polygons = []
    # print(len(p))
    # for pi in range(len(p)):
    #     polygon = p[pi].vertices
    #     #check to make sure the polygon is a closed loop
    #     if polygon[-1,0] == polygon[0,0] and polygon[-1,1] == polygon[0,1]:
    #         polygons.append(polygon)
    # for col in cs.collections:
    # # Loop through all polygons that have the same intensity level
    print(len(p))
    # print(p)
    for contour_path in p: 
        # Create the polygon for this intensity level
        # The first polygon in the path is the main one, the following ones are "holes"
        if contour_path.to_polygons():
            print(contour_path.to_polygons())
            # print(len(contour_path.to_polygons()))
            for ncp,cp in enumerate(contour_path.to_polygons()):
                print(cp)
                cp = np.array(cp)
                x = cp[:,0]
                y = cp[:,1]
                if ncp>0:
                    print('multiple contour paths detected')
                # shapely implicitly closes any linear rings that don't have matching first/last vertices
                new_shape = shapely.geometry.Polygon([(i[0], i[1]) for i in zip(x,y)])
                # if new_shape.is_valid:
                #     poly = new_shape
                # else:
                #     print(new_shape)
                #     poly = new_shape.buffer(0)
                #     print(poly)
                
                if ncp == 0:
                    poly = new_shape
                    # print('had multiple contour paths')
                else:
                    # Remove the holes if there are any
                    poly = new_shape.difference(new_shape)
                    # print('had to remove a hole')
                    # Can also be left out if you want to include all rings
                # print(len(poly.exterior.coords))

                polygons.append(list(poly.exterior.coords))

                # polygon = poly.vertices
                # #check to make sure the polygon is a closed loop
                # if polygon[-1,0] == polygon[0,0] and polygon[-1,1] == polygon[0,1]:
                #     polygons.append(polygon)
        else:
            print('one did not evaluate!')

    print(len(polygons))
    print(type(polygons))

    plt.close(fig)

    return polygons
