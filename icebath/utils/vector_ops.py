import warnings
import numpy as np

# can likely depricate this function (use built-in area methods of geospatial libraries instead)
def get_poly_area(outlinePolygonSet):
    '''
    Given a true-scale set of polygon vertices, compute the area of each polygon.

    From Mike Wood's get_surface_areas

    Example
    -------
    >>> areas = vector_ops.get_poly_area(ds['berg_outlines'].values[timei])
    '''

    areas=[]
    
    #DevGoal: use gdal/ogr to complete this calculation and account for crs
    warnings.warn("You MUST supply the polygon vertices in a true scale coordinate system!")
    
    for polygon in outlinePolygonSet:
        area = 0.5 * np.abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
        areas.append(area)
    return(areas)

def poly_complexity(geom):
    """
    Compute a polygon's complexity. Based on: Brinkhoff et al (white paper?)
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.1045&rep=rep1&type=pdf

    Implementation updated (to Python >=3 and for shapely) and modified from QGIS 2.0 plugin:
    https://github.com/pondrejk/PolygonComplexity
    Original code was more generalized and handled multipologons, etc. In updating it (especially
    the geospatial/shapely portions) I also de-generalized it, so it now doesn't handle polygons
    with multiple rings or multipolygons.

    Parameters
    ----------
    geom : shapely Polygon geometry object
    """

    # feature_compactness = find_feature_compactness(geom)
    feature_convexity = find_feature_convexity(geom)
    feature_amplitude = find_feature_amplitude(geom)
    feature_vertices = find_feature_vertices(geom)
    feature_notches = find_feature_notches(geom)
    feature_notches_normalized = float(feature_notches) / (feature_vertices - 3)
    feature_frequency = (16*((feature_notches_normalized - 0.5)**4)) - (8*(feature_notches_normalized - 0.5)**2) + 1
    feature_complexity = find_feature_complexity(feature_convexity, feature_amplitude, feature_frequency)
    
    return feature_complexity

def find_feature_compactness(geom):
    return (geom.length/(3.54 * np.sqrt(geom.area))) # 1 for a circle

def find_feature_convexity(geom):
    # hull_area = 0
    # geom_area = 0
    hull_area = geom.convex_hull.area
    return (hull_area - geom.area)/hull_area

def find_feature_amplitude(geom):
    # hull_length = 0
    # geom_length = 0
    hull_length =  geom.convex_hull.length
    geom_length =  geom.length
    return (geom_length - hull_length)/geom_length

def find_feature_notches(geom):
    if geom is None: return None
    if geom.type == 'Polygon':
        notches = 0  
        ringlist = list(geom.exterior.coords)
        ringlist.append(ringlist[1])
        triplet = []

        for i in ringlist:
            triplet.append(i) 
            if len(triplet) > 3:
                del triplet[0]
            if len(triplet) == 3:
                zcp = find_convex(triplet)
                if zcp > 0: 
                    notches +=1

    return notches

def find_feature_vertices(geom):
    if geom is None: return None
    # if geom.type == 'Polygon':
    #     count = 0
    count = len(list(geom.exterior.coords)) - 1.0
    return count

def find_feature_complexity(conv, ampl, freq):
    return ((0.8 * ampl * freq) + (0.2 * conv))

def find_convex(triplet):
    a1,a2,a3 = triplet[0], triplet[1], triplet[2]
    dx1 = a2[0] - a1[0]
    dy1 = a2[1] - a1[1]
    dx2 = a3[0] - a2[0]
    dy2 = a3[1] - a2[1]
    zcrossproduct = dx1*dy2 - dy1*dx2
    return zcrossproduct