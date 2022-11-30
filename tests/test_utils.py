# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from orbit_defender2d.utils import utils as U

from hypothesis import given, strategies as st

@given(st.integers(min_value=0))
def test_hypothesis_int2bitlist(n):
    '''test int to binary conversion'''
    n_bit = U.int2bitlist(n)
    # reconstruct integer from binary
    n_int = 0
    for i, v in enumerate(reversed(n_bit)):
        n_int += v * 2**i
    assert n_int == n

@given(st.integers(min_value=0))
def test_hypothesis_int2bitlist_bitlist2in(n):
    '''test int to binary and binary to int conversion'''
    assert U.bitlist2int(U.int2bitlist(n)) == n
