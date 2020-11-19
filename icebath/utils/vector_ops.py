import warnings
import numpy as np

def get_poly_area(outlinePolygonSet):
    '''
    Given a true-scale set of polygon vertices, compute the area of each polygon.

    From Mike Wood's get_surface_areas
    '''

    areas=[]
    
    #DevGoal: use gdal/ogr to complete this calculation and account for crs
    warnings.warn("You MUST supply the polygon vertices in a true scale coordinate system!")
    
    for polygon in outlinePolygonSet:
        area = 0.5 * np.abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
        areas.append(area)
    return(areas)

