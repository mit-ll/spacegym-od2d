# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest
import orbit_defender2d.utils.utils as U
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from orbit_defender2d.utils.engagement_graph import EngagementGraph
from orbit_defender2d.utils.utils import EngagementTuple as ET


ENG_CASE_0 = {'alpha:seeker:0': ET('noop', 'alpha:seeker:0', 1.0),
     'beta:seeker:0': ET('noop', 'beta:seeker:0', 1.0),
     'beta:seeker:1': ET('guard', 'beta:seeker:0', 1.0)}

ENG_CASE_1 = {'alpha:seeker:0': ET('noop', 'alpha:seeker:0', 1.0),
     'beta:seeker:0': ET('noop', 'beta:seeker:0', 1.0)}

ENG_CASE_2 = {'alpha:seeker:0': ET('noop', 'alpha:seeker:0', 1.0),
     'beta:seeker:0': ET('collide', 'alpha:seeker:0', 1.0),
     'beta:seeker:1': ET('collide', 'alpha:seeker:0', 1.0)}

ENG_CASE_3 = {'alpha:bludger:0': ET('noop', 'alpha:bludger:0', 1.0),
     'alpha:seeker:0': ET('guard', 'alpha:bludger:0', 1.0),
     'alpha:seeker:1': ET('noop', 'alpha:seeker:1', 1.0),
     'beta:seeker:0': ET('shoot', 'alpha:bludger:0', 1.0)}

# ENG_CASE_2 = {'alpha:seeker:0': (0, 'alpha:seeker:0'),
#      'beta:seeker:0': (0, 'beta:seeker:0'),
#      'beta:seeker:1': ('guard', 'beta:seeker:0')}

class Foo:
    def __init__(self):
        pass

def test_EngagementGraph_init_0():
    p1p0 = 'alpha:seeker:0'
    p1p1 = 'alpha:bludger:1'
    p2p0 = 'beta:seeker:0'
    p2p1 = 'beta:bludger:1'
    engagements = dict()
    engagements[p1p0] = ET('noop', p1p0, 1.0)
    engagements[p1p1] = ET('shoot', p2p0, 1.0)
    engagements[p2p0] = ET('noop', p2p0, 1.0)
    engagements[p2p1] = ET('guard', p2p0, 1.0)
    eg = EngagementGraph(engagements)


partial_piece_naming_st = st.tuples(
    st.sampled_from([U.SEEKER, U.BLUDGER]),
    st.integers(min_value=0, max_value=1e3)).map(lambda x: x[0] + ":"+str(x[1]))

p1_naming_st = st.sets(
    partial_piece_naming_st, 
    min_size=1, max_size=10).map(lambda x: [U.P1 + ':' + elem for elem in x])

p2_naming_st = st.sets(
    partial_piece_naming_st, 
    min_size=1, max_size=10).map(lambda x: [U.P2 + ':' + elem for elem in x])

@st.composite
def engagement_w_prob1_pattern_st(draw):
    p1_pieces = draw(p1_naming_st)
    n_p1 = len(p1_pieces)
    p2_pieces = draw(p2_naming_st)
    n_p2 = len(p2_pieces)
    eng_types = [U.NOOP]+U.ENGAGEMENT_TYPES
    reduced_eng_types = [U.NOOP, U.SHOOT, U.COLLIDE]
    eng_prob = 1.0

    engagements = dict()
    for piece in p1_pieces:
        # p1_eng_type = np.random.choice(eng_types) if n_p1 > 1 else np.random.choice(reduced_eng_types)
        if n_p1 > 1:
            p1_eng_type = draw(st.sampled_from(eng_types))
        else:
            p1_eng_type = draw(st.sampled_from(reduced_eng_types))
        p1_target = None
        if p1_eng_type in [str(U.NOOP), U.NOOP]:
            p1_eng_type = U.NOOP
            p1_target = piece
        elif p1_eng_type in [U.SHOOT, U.COLLIDE]:
            p1_target = np.random.choice(p2_pieces)
        elif p1_eng_type == U.GUARD:
            p1_target = np.random.choice(np.setdiff1d(p1_pieces, piece))
        else:
            raise ValueError('Invalid engagment type {}'.format(p1_eng_type))
        engagements[piece] = ET(p1_eng_type, p1_target, eng_prob)

    for piece in p2_pieces:
        # p2_eng_type = np.random.choice(eng_types) if n_p2 > 1 else np.random.choice(reduced_eng_types)
        if n_p2 > 1:
            p2_eng_type = draw(st.sampled_from(eng_types))
        else:
            p2_eng_type = draw(st.sampled_from(reduced_eng_types))
        p2_target = None
        if p2_eng_type in [str(U.NOOP), U.NOOP]:
            p2_eng_type = U.NOOP
            p2_target = piece
        elif p2_eng_type in [U.SHOOT, U.COLLIDE]:
            p2_target = np.random.choice(p1_pieces)
        elif p2_eng_type == U.GUARD:
            p2_target = np.random.choice(np.setdiff1d(p2_pieces, piece))
        else:
            raise ValueError('Invalid engagment type {}'.format(p1_eng_type))
        engagements[piece] = ET(p2_eng_type, p2_target, eng_prob)

    return engagements

@given(engagement_w_prob1_pattern_st())
def test_hypothesis_EngagementGraph_init(engagements):
    eg = EngagementGraph(engagements)

def test_EngagementGraph_resolve_guard_engagements_case0():
    # ~~~ ARRANGE ~~~
    eg = EngagementGraph(engagements=ENG_CASE_0)

    # ~~~ ACT ~~~
    eg_guard_out = eg.resolve_guard_engagements()

    # ~~~ ASSERT ~~~
    # should only be one outcome from the trivial guard
    assert len(eg_guard_out) == 1
    triv_guard = eg_guard_out[0]
    assert triv_guard.action_type == U.GUARD
    assert triv_guard.attacker is None
    assert triv_guard.success is False

@given(engagement_w_prob1_pattern_st())
def test_hypothesis_EngagementGraph_resolve_guard_engagements(engagements):
    eg = EngagementGraph(engagements)
    n_old_tot_edges = len(eg.egraph.edges)
    n_old_guard_edges = len([ge for ge in eg.egraph.edges(data=True) if ge[2]['etype']==U.GUARD])
    n_old_shoot_edges = len([ge for ge in eg.egraph.edges(data=True) if ge[2]['etype']==U.SHOOT])
    n_old_collide_edges = len([ge for ge in eg.egraph.edges(data=True) if ge[2]['etype']==U.COLLIDE])

    # resolve guard engagments with probability 1
    eg.resolve_guard_engagements()
    n_new_tot_edges = len(eg.egraph.edges)
    n_new_guard_edges = len([ge for ge in eg.egraph.edges(data=True) if ge[2]['etype']==U.GUARD])
    n_new_shoot_edges = len([ge for ge in eg.egraph.edges(data=True) if ge[2]['etype']==U.SHOOT])
    n_new_collide_edges = len([ge for ge in eg.egraph.edges(data=True) if ge[2]['etype']==U.COLLIDE])

    # ensure total number of edges has decreased by the number of guard edges
    assert n_new_guard_edges == 0
    assert n_new_tot_edges == n_old_tot_edges - n_old_guard_edges

    # ensure the number of collide and shoot edges has remained the same
    assert n_old_shoot_edges == n_new_shoot_edges
    assert n_old_collide_edges == n_new_collide_edges

@given(engagement_w_prob1_pattern_st())
def test_hypothesis_EngagementGraph_resolve_shoot_engagements(engagements):
    eg = EngagementGraph(engagements)
    n_old_tot_nodes = len(eg.egraph.nodes)
    old_shoot_tar_set = set([node for node in eg.egraph.nodes() if any([dat['etype']==U.SHOOT for _, _, dat in eg.egraph.in_edges(node, data=True)])])
    # n_old_shoot_tar = len(old_shoot_tar_set)

    # resolve guard engagments with probability 1
    eg.resolve_shoot_engagements()
    n_new_tot_nodes = len(eg.egraph.nodes)
    n_new_shoot_edges = len([ge for ge in eg.egraph.edges(data=True) if ge[2]['etype']==U.SHOOT])

    # check that all shoot engagements resolved
    assert n_new_shoot_edges == 0

    # for prob=1.0, check that number of removed nodes is expected
    assert n_new_tot_nodes == n_old_tot_nodes - len(old_shoot_tar_set)

@given(engagement_w_prob1_pattern_st())
def test_hypothesis_EngagementGraph_resolve_collide_engagements(engagements):
    eg = EngagementGraph(engagements)
    n_old_tot_nodes = len(eg.egraph.nodes)
    # old_collide_tar_set = set([node for node in eg.egraph.nodes() if any([dat['etype']==U.COLLIDE for _, _, dat in eg.egraph.in_edges(node, data=True)])])
    # old_collide_src_set = set([node for node in eg.egraph.nodes() if any([dat['etype']==U.COLLIDE for _, _, dat in eg.egraph.out_edges(node, data=True)])])
    # old_collide_node_set = old_collide_src_set.union(old_collide_tar_set)
    # n_old_shoot_tar = len(old_shoot_tar_set)

    # resolve guard engagments with probability 1
    col_out = eg.resolve_collide_engagements()
    n_collisions = len(col_out)
    n_new_tot_nodes = len(eg.egraph.nodes)
    n_new_collide_edges = len([ge for ge in eg.egraph.edges(data=True) if ge[2]['etype']==U.COLLIDE])

    # check that all shoot engagements resolved
    assert n_new_collide_edges == 0

    # for prob=1.0, check that number of removed nodes is expected
    assert n_new_tot_nodes == n_old_tot_nodes - 2 * n_collisions

@given(engagement_w_prob1_pattern_st())
def test_hypothesis_EngagementGraph_resolve_all_engagements(engagements):
    eg = EngagementGraph(engagements)
    eg.resolve_guard_engagements()
    eg.resolve_shoot_engagements()
    eg.resolve_collide_engagements()
    assert len(eg.egraph.edges) == 0

if __name__ == '__main__':
    # test_EngagementGraph_init_0()
    # engagement_pattern_st().example()
    test_hypothesis_EngagementGraph_resolve_guard_engagements(ENG_CASE_0)
    # test_hypothesis_EngagementGraph_resolve_collide_engagements(ENG_CASE_2)
    # test_hypothesis_EngagementGraph_resolve_all_engagements(ENG_CASE_3)
