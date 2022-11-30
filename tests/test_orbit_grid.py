# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from orbit_defender2d.utils.orbit_grid import OrbitGrid
import pytest
from hypothesis import given, strategies as st

sector_cases = [
    (0, (0,0)),
    (1, (1,0)),
    (2, (1,1)),
    (3, (2,0)),
    (4, (2,1)),
    (5, (2,2)),
    (6, (2,3)),
    (7, (3,0)),
    (8, (3,1)),
    (9, (3,2)),
    (10, (3,3)),
    (11, (3,4)),
    (12, (3,5)),
    (13, (3,6)),
    (14, (3,7)),
    (15, (4,0))]

@st.composite
def orbit_grid_init_st(draw):
    n_rings = draw(st.integers(min_value=1, max_value=1000))
    sector = draw(
        st.integers(min_value=0, max_value=2**(n_rings+1)-2))
    return (n_rings, sector)

@pytest.mark.parametrize(
    "sec_num,exp_coord",sector_cases)
def test_sector_num2ring(sec_num, exp_coord):
    og = OrbitGrid(n_rings=4)
    ring = og.sector_num2ring(sec_num)
    assert ring == exp_coord[0]

@pytest.mark.parametrize(
    "sec_num,exp_coord",sector_cases)
def test_sector_num2coord(sec_num, exp_coord):
    og = OrbitGrid(n_rings=4)
    sec_coord = og.sector_num2coord(sec_num)
    assert sec_coord == exp_coord

@given( orbit_grid_init_st() )
def test_hypothesis_num2coord_coord2num(ring_sec):
    n_rings = ring_sec[0]
    sec_num = ring_sec[1]
    og = OrbitGrid(n_rings=n_rings)
    ring, azim = og.sector_num2coord(sec_num=sec_num)
    assert og.sector_coord2num(ring, azim) == sec_num

@given( orbit_grid_init_st() )
def test_hypothesis_get_radial_out_in_sector(ring_sec):
    n_rings = ring_sec[0]
    sec_num = ring_sec[1]
    og = OrbitGrid(n_rings=n_rings)
    ring, azim = og.sector_num2coord(sec_num=sec_num)
    out_sec_num = og.get_radial_out_sector(sec_num=sec_num)
    if ring < n_rings:
        assert og.get_radial_in_sector(out_sec_num) == sec_num
    else:
        assert out_sec_num is None

@pytest.mark.parametrize(
    "n_rings,sec_num,exp_out_sec_num", [
        (1, 0, 1),
        (2, 0, 1),
        (2, 1, 3),
        (2, 2, 5),
        (3, 0, 1),
        (3, 1, 3),
        (3, 2, 5),
        (3, 3, 7),
        (3, 4, 9),
        (3, 5, 11),
        (3, 6, 13)])
def test_get_radial_out_sector(n_rings, sec_num, exp_out_sec_num):
    og = OrbitGrid(n_rings=n_rings)
    out_sec_num = og.get_radial_out_sector(sec_num)
    assert out_sec_num == exp_out_sec_num

@pytest.mark.parametrize(
    "n_rings,sec_num,exp_in_sec_num", [
        (1, 1, 0),
        (1, 2, 0),
        (2, 1, 0),
        (2, 2, 0),
        (2, 3, 1),
        (2, 4, 1),
        (2, 5, 2),
        (2, 6, 2),
        (3, 1, 0),
        (3, 2, 0),
        (3, 3, 1),
        (3, 4, 1),
        (3, 5, 2),
        (3, 6, 2),
        (3, 7, 3),
        (3, 8, 3),
        (3, 9, 4),
        (3, 10, 4),
        (3, 11, 5),
        (3, 12, 5),
        (3, 13, 6),
        (3, 14, 6)])
def test_get_radial_in_sector(n_rings, sec_num, exp_in_sec_num):
    og = OrbitGrid(n_rings=n_rings)
    in_sec_num = og.get_radial_in_sector(sec_num)
    assert in_sec_num == exp_in_sec_num

@pytest.mark.parametrize(
    "n_rings,sec_num,rel_azim,exp_new_sec_num", [
        (1, 1, 1, 2),
        (1, 1, -1, 2),
        (1, 1, 2, 1),
        (1, 1, -2, 1),
        (1, 1, 100, 1),
        (1, 2, 1, 1),
        (1, 2, -1, 1),
        (1, 2, 2, 2),
        (1, 2, -2, 2),
        (1, 2, 100, 2),
        (2, 1, 1, 2),
        (2, 1, -1, 2),
        (2, 1, 2, 1),
        (2, 1, -2, 1),
        (2, 1, 100, 1),
        (2, 2, 1, 1),
        (2, 2, -1, 1),
        (2, 2, 2, 2),
        (2, 2, -2, 2),
        (2, 2, 100, 2),
        (3, 7, 1, 8),
        (3, 7, -1, 14),
        (3, 6, -2, 4),
        (3, 7, 0, 7),
        (3, 8, 0, 8),
        (3, 9, 0, 9),
        (3, 10, 0, 10),
        (3, 11, 0, 11),
        (3, 12, 0, 12),
        (3, 13, 0, 13),
        (3, 14, 0, 14),
    ])
def test_get_relative_azimuth_sector(n_rings, sec_num, rel_azim, exp_new_sec_num):
    og = OrbitGrid(n_rings=n_rings)
    new_sec_num = og.get_relative_azimuth_sector(sec_num=sec_num, rel_azim=rel_azim)
    assert new_sec_num == exp_new_sec_num

@given( orbit_grid_init_st(), st.integers(min_value=-1e3, max_value=1e3) )
def test_hypothesis_get_relative_azimuth_sector(ring_sec, rel_azim):
    # generate orbit grid
    n_rings = ring_sec[0]
    sec_num = ring_sec[1]
    og = OrbitGrid(n_rings=n_rings)

    # get new sector number based on relative azimuth
    new_sec_num = og.get_relative_azimuth_sector(sec_num=sec_num, rel_azim=rel_azim)

    # check that new sector is in same ring
    old_sec_ring = og.sector_num2ring(sec_num=sec_num)
    new_sec_ring = og.sector_num2ring(sec_num=new_sec_num)
    assert old_sec_ring == new_sec_ring

@pytest.mark.parametrize(
    "n_rings,sec_num,exp_adj_secs", [
        (1, 1, [2,0]),
        (1, 2, [1,0]),
        (2, 1, [2,0,3,4]),
        (2, 2, [1,0,5,6]),
        (2, 3, [4,6,1]),
        (2, 4, [5,3,1]),
        (2, 5, [6,4,2]),
        (2, 6, [3,5,2]),
        (3, 1, [2,0,3,4]),
        (3, 2, [1,0,5,6]),
        (3, 3, [4,6,1,7,8]),
        (3, 4, [5,3,1,9,10]),
        (3, 5, [6,4,2,11,12]),
        (3, 6, [3,5,2,13,14]),
        (3, 7, [8,14,3]),
        (3, 8, [9,7,3]),
        (3, 9, [10,8,4]),
        (3, 10, [11,9,4]),
        (3, 11, [12,10,5]),
        (3, 12, [13,11,5]),
        (3, 13, [12,14,6]),
        (3, 14, [7,13,6]),
        ])
def test_get_all_adjacent_sectors(n_rings, sec_num, exp_adj_secs):
    og = OrbitGrid(n_rings=n_rings)
    adj_secs= og.get_all_adjacent_sectors(sec_num=sec_num)
    assert adj_secs == set(exp_adj_secs)

if __name__ == "__main__":
    test_get_radial_in_sector(3, 4, 1)
