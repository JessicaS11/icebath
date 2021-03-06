def get_sw_dens(fjord):
    """
    Get the fjord-specific seawater density, in kg/m3
    """
    rho_sw = {"JI":1027.3, "UP":1028.5}
    rho_sw_err = {"JI":1, "UP":1}

    try:
        return [rho_sw.pop(fjord), rho_sw_err.pop(fjord)]
    except KeyError:
        "The current fjord does not have a seawater density entry"


def get_mouth_coords(fjord):
    """
    Get the reference x and y coordinates for the fjord mouth
    """

    x = {"JI": -312319.963189}
    y = {"JI": -2260417.83078}

    try:
        return [x.pop(fjord), y.pop(fjord)]
    except KeyError:
        "The current fjord does not have a location entry"


def get_min_berg_area(fjord):
    """
    Get a minimum iceberg area for the given fjord
    """

    area = {"JI": 50000, "UP":4000}

    try:
        return area.pop(fjord)
    except KeyError:
        "The current fjord does not have a location entry - using a default value!"
        return 5000


def get_ice_thickness(fjord):
    """
    Get the approximate ice thickness (in m) at the grounding line for a given fjord
    """

    thickness = {"JI": 1500}

    try:
        return thickness.pop(fjord)
    except KeyError:
        "The current fjord does not have a location entry - using a default value!"
        return 1500



