import pytest
import warnings

import fjord_props as fjord

def test_get_seawater_props():
    obs = fjord.get_sw_dens('JI')
    exp = [1027.3, 1]
    assert obs == exp
    
def test_invalid_fjord():
    ermsg = "The current fjord does not have a seawater density entry"
    with pytest.raises(TypeError, match=ermsg):
        fjord.get_sw.dens('QQ')
