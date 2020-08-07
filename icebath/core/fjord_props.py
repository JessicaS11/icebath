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
