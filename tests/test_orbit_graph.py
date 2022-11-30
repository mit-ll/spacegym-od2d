# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest
from hypothesis import given, strategies as st
from orbit_defender2d.utils.orbit_grid import OrbitGrid
from orbit_defender2d.utils.orbit_graph import OrbitGraph


board_cases = [
    (2),
    (3),
    (4)]

@pytest.mark.parametrize(
    "num_rings",board_cases)
def test_orbit_graph_nodes(num_rings):
    ogrid = OrbitGrid(num_rings)
    ograph = OrbitGraph(ogrid)
    assert ogrid.n_sectors == len(list(ograph.adj_graph))
    assert ogrid.n_sectors == len(list(ograph.move_graph))
    assert ogrid.n_sectors == len(list(ograph.turn_graph))


#check edge degree count

    
