def get_sw_dens(fjord):
    """
    Get the fjord-specific seawater density, in kg/m3
    For JI and UP, the density is computed at 250 m depth
    For KB, the density is computed at 150 m depth
    """
    rho_sw = {"JI":1027.3, "UP":1028.5, "KB":1027.9}
    rho_sw_err = {"JI":1, "UP":1, "KB":1}

    try:
        return [rho_sw.pop(fjord), rho_sw_err.pop(fjord)]
    except KeyError:
        print("The current fjord does not have a seawater density entry - using a default")
        return 1028.0, 1

def get_mouth_coords(fjord):
    """
    Get the reference x and y coordinates for the fjord mouth
    """

    x = {"JI": -312319.963189, "KB": -438759}
    y = {"JI": -2260417.83078, "KB": -1037230}

    try:
        return [x.pop(fjord), y.pop(fjord)]
    except KeyError:
        print("The current fjord does not have a location entry")


def get_min_berg_area(fjord):
    """
    Get a minimum iceberg area for the given fjord
    """

    area = {"JI": 50000, "UP":4000}

    try:
        return area.pop(fjord)
    except KeyError:
        print("The current fjord does not have a minimum berg size entry - using a default value!")
        return 5000


def get_ice_thickness(fjord):
    """
    Get the approximate ice thickness (in m) at the grounding line for a given fjord
    """

    # Note: these values are approximated manually by inspecting the ice thickness layer in BedMachine v3 ~10 km in from the terminus
    # and choosing a value at the upper end of gridded values.
    thickness = {"JI": 1500, "KB": 800}

    try:
        return thickness.pop(fjord)
    except KeyError:
        print("The current fjord does not have an ice thickness entry - using a default value!")
        return 1500

def get_fjord_bounds(fjord):
    """
    Get geospatial bounds of the fjord to subset measurement files (e.g. BedMachine, IBCAO)
    Coordinates are in NSIDC Polar Geospatial Coordinates (EPSG:3413)
    """
    
    minx = {"JI": -440795.0, "KB": -456119.7}
    maxx = {"JI": -144209.6, "KB": -364443.8}
    miny = {"JI": -2354791.7, "KB": -1104271.1}
    maxy = {"JI": -2076077.5, "KB": -1007037.0}

    try:
        return (minx.pop(fjord), miny.pop(fjord), maxx.pop(fjord), maxy.pop(fjord))
    except KeyError:
        print("The current fjord does not have a bounding box entry")

