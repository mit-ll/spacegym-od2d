# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest
import time
import numpy as np
import orbit_defender2d.utils.utils as U
from pettingzoo.utils import wrappers
from orbit_defender2d.king_of_the_hill import koth
from hypothesis import given, settings, Verbosity
from hypothesis import strategies as st
from orbit_defender2d.utils.utils import EngagementTuple as ET
from orbit_defender2d.utils.utils import MovementTuple as MT
from orbit_defender2d.king_of_the_hill import default_game_parameters as DGP
from copy import deepcopy

P1S0 = 'Alpha:HVA:0'
P1B1 = 'Alpha:Patrol:1'
P2S0 = 'Bravo:HVA:0'
P2B1 = 'Bravo:Patrol:1'

INIT_BOARD_PATTERN_0 = [(-2,1), (-1,3), (0,2), (1,3), (2,1)] 
INIT_BOARD_PATTERN_1 = [] 
INIT_BOARD_PATTERN_2 = [(0,1)] 

DEFAULT_PARAMS_PARTIAL = {
    'init_fuel' : DGP.INIT_FUEL,
    'init_ammo' : DGP.INIT_AMMO,
    'min_fuel' : DGP.MIN_FUEL,
    'fuel_usage' : DGP.FUEL_USAGE,
    'engage_probs' : DGP.ENGAGE_PROBS,
    'illegal_action_score' : DGP.ILLEGAL_ACT_SCORE,
    'in_goal_points' : DGP.IN_GOAL_POINTS,
    'adj_goal_points' : DGP.ADJ_GOAL_POINTS,
    'fuel_points_factor': DGP.FUEL_POINTS_FACTOR,
    'win_score' : DGP.WIN_SCORE,
    'max_turns' : DGP.MAX_TURNS,
    'fuel_points_factor_bludger': DGP.FUEL_POINTS_FACTOR_BLUDGER,
    }

DEP_MOVE_SEQ_0 = [
    {U.P1: [U.RADIAL_OUT], U.P2: [U.NOOP]},
    {U.P1: [U.PROGRADE], U.P2: [U.NOOP]},
    {U.P1: [U.PROGRADE], U.P2: [U.NOOP]},
    {U.P1: [U.PROGRADE], U.P2: [U.NOOP]},
    {U.P1: [U.RADIAL_IN], U.P2: [U.NOOP]}]

MOVE_SEQ_0 = [
    {P1S0: MT(U.RADIAL_OUT), P2S0: MT(U.NOOP)},
    {P1S0: MT(U.PROGRADE), P2S0: MT(U.NOOP)},
    {P1S0: MT(U.PROGRADE), P2S0: MT(U.NOOP)},
    {P1S0: MT(U.PROGRADE), P2S0: MT(U.NOOP)},
    {P1S0: MT(U.RADIAL_IN), P2S0: MT(U.NOOP)},
]

EXP_POS_0 = [(15, 23), (31, 23), (32, 23), (33, 23), (34, 23), (16, 23)]

# ENGAGE_0 = {'Alpha': [('shoot', 1), ('guard', 0)], 'Bravo': [(0,), ('collide', 0)]}
ENGAGE_0 = {
    P1S0: ET('noop', 'Alpha:HVA:0', 1.0),
    P1B1: ET('shoot', 'Bravo:HVA:0', 1.0),
    P2S0: ET('noop', 'Bravo:HVA:0', 1.0),
    P2B1: ET('guard', 'Bravo:HVA:0', 1.0),
}

ordered_integer_triplet_st = st.lists(
    st.integers(min_value=1, max_value=10),
    min_size=3,
    max_size=3).map(lambda x: tuple(sorted(x)))

game_init_pattern_st = st.lists(
    st.tuples(st.integers(min_value=-1e4, max_value=1e4), st.integers(min_value=0, max_value=16)),
    max_size=16)

@st.composite
def game_state_st(draw):
    '''draw a valid positions, fuel, and ammo for a valid game state'''

    # draw valid rings for orbit grid
    ring_def = draw(ordered_integer_triplet_st)
    max_ring = ring_def[2]
    min_ring = ring_def[0]
    geo_ring = ring_def[1]

    # draw valid game init
    init_pattern = draw(game_init_pattern_st)

    # create game object
    game = koth.KOTHGame(
        max_ring=max_ring, 
        min_ring=min_ring, 
        geo_ring=geo_ring,
        init_board_pattern_p1=init_pattern,
        init_board_pattern_p2=init_pattern,
        # **DEFAULT_PARAMS_PARTIAL)
        init_fuel=DGP.INIT_FUEL,
        init_ammo=DGP.INIT_AMMO,
        min_fuel=DGP.MIN_FUEL,
        fuel_usage=DGP.FUEL_USAGE,
        engage_probs=DGP.ENGAGE_PROBS,
        illegal_action_score=DGP.ILLEGAL_ACT_SCORE,
        in_goal_points=DGP.IN_GOAL_POINTS,
        adj_goal_points=DGP.ADJ_GOAL_POINTS,
        fuel_points_factor=DGP.FUEL_POINTS_FACTOR,
        win_score=DGP.WIN_SCORE,
        max_turns=DGP.MAX_TURNS,
        fuel_points_factor_bludger=DGP.FUEL_POINTS_FACTOR_BLUDGER,)

    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=init_pattern, init_pattern_beta=init_pattern)

    # n_tokens = game.n_tokens_alpha + game.n_tokens_beta
    min_sector = game.board_grid.sector_coord2num(game.inargs.min_ring, 0)
    max_sector = game.board_grid.n_sectors-1

    # generate list of valid fuel amounts
    fuels_alpha = draw(st.lists(
        elements=st.floats(min_value=0.0, max_value=100.0, allow_infinity=False, allow_nan=False),
        min_size=game.n_tokens_alpha, max_size=game.n_tokens_alpha))
    fuels_beta = draw(st.lists(
        elements=st.floats(min_value=0.0, max_value=100.0, allow_infinity=False, allow_nan=False),
        min_size=game.n_tokens_beta, max_size=game.n_tokens_beta))

    # generate list of valid position
    positions_alpha = draw(st.lists(
        elements=st.integers(min_value=min_sector, max_value=max_sector),
        min_size=game.n_tokens_alpha, max_size=game.n_tokens_alpha))
    positions_beta = draw(st.lists(
        elements=st.integers(min_value=min_sector, max_value=max_sector),
        min_size=game.n_tokens_beta, max_size=game.n_tokens_beta))

    # generate list of valid ammo
    ammo_alpha = draw(st.lists(
        elements=st.integers(min_value=0, max_value=1),
        min_size=game.n_tokens_alpha, max_size=game.n_tokens_alpha))
    ammo_beta = draw(st.lists(
        elements=st.integers(min_value=0, max_value=1),
        min_size=game.n_tokens_beta, max_size=game.n_tokens_beta))

    # encode into game state
    for i in range(game.n_tokens_alpha):
        game.game_state[U.P1][U.TOKEN_STATES][i].position = positions_alpha[i]
        game.game_state[U.P1][U.TOKEN_STATES][i].satellite.fuel = fuels_alpha[i]
        game.game_state[U.P1][U.TOKEN_STATES][i].satellite.ammo = ammo_alpha[i]
    for i in range(game.n_tokens_beta):
        game.game_state[U.P2][U.TOKEN_STATES][i].position = positions_beta[i]
        game.game_state[U.P2][U.TOKEN_STATES][i].satellite.fuel = fuels_beta[i]
        game.game_state[U.P2][U.TOKEN_STATES][i].satellite.ammo = ammo_beta[i]

    # generate valid turn phase and turn count
    game.game_state[U.TURN_COUNT] = draw(st.integers(min_value=0, max_value=1e6))
    turn_phase = draw(st.sampled_from(U.TURN_PHASE_LIST))
    game.update_turn_phase(turn_phase)

    # generate valid scores
    game.game_state[U.P1][U.SCORE] = draw(st.floats(min_value=-1e3, max_value=1e3, allow_infinity=False, allow_nan=False))
    game.game_state[U.P2][U.SCORE] = draw(st.floats(min_value=-1e3, max_value=1e3, allow_infinity=False, allow_nan=False))


    # return new token game state
    return game


@given(game_state_st())
def test_hypothesis_KOTHGame_game_state(game):
    # check state reprsentation equal 
    for player in game.player_names:
        n_player_tokens = game.n_tokens_alpha if player==U.P1 else game.n_tokens_beta
        for sat_i in range(n_player_tokens):
            token_type = U.SEEKER if sat_i == 0 else U.BLUDGER
            token_name = player + ':' + token_type + ':' + str(sat_i)
            assert game.token_catalog[token_name].position == game.game_state[player][U.TOKEN_STATES][sat_i].position
            assert game.token_catalog[token_name].satellite.fuel == game.game_state[player][U.TOKEN_STATES][sat_i].satellite.fuel
            assert game.token_catalog[token_name].satellite.ammo == game.game_state[player][U.TOKEN_STATES][sat_i].satellite.ammo

@given(game_state_st())
def test_hypothesis_KOTHGame_terminate_game(game):
    # check game is terminated with zero-sum result
    pre_term_score = (game.game_state[U.P1][U.SCORE], game.game_state[U.P2][U.SCORE])
    res = game.terminate_game()
    assert game.game_state[U.GAME_DONE]
    assert np.isclose(res[U.P1], -res[U.P2])
    # post_term_score = (game.game_state[U.P1][U.SCORE], game.game_state[U.P2][U.SCORE])
    # assert np.isclose(
    #     pre_term_score[0] + game.inargs.fuel_points_factor*game.game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel,
    #     post_term_score[0])
    # assert np.isclose(
    #     pre_term_score[1] + game.inargs.fuel_points_factor*game.game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel,
    #     post_term_score[1])


def test_KOTHGame_initial_game_state():
    koth_game = koth.KOTHGame(
        max_ring = 3,
        min_ring = 1,
        geo_ring = 3,
        init_board_pattern_p1=INIT_BOARD_PATTERN_0,
        init_board_pattern_p2=INIT_BOARD_PATTERN_0,
        **DEFAULT_PARAMS_PARTIAL)
    # bs, piece_cat, n_tokens_alpha, n_tokens_beta = \
    #     koth_game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_0, init_pattern_beta=INIT_BOARD_PATTERN_0)
    bs = koth_game.game_state
    assert koth_game.n_tokens_alpha == 11
    assert koth_game.n_tokens_beta == 11
    assert bs['goal_sector_alpha'] == 7
    assert bs['goal_sector_beta'] == 11
    assert len(bs['Alpha']['token_states']) == 11
    assert len(bs['Bravo']['token_states']) == 11
    assert bs['Alpha']['token_states'][0].role == 'HVA'
    assert bs['Alpha']['token_states'][0].position == 7
    assert bs['Alpha']['token_states'][1].role == 'Patrol'
    assert bs['Alpha']['token_states'][1].position == 13
    assert bs['Alpha']['token_states'][2].role == 'Patrol'
    assert bs['Alpha']['token_states'][2].position == 14
    assert bs['Alpha']['token_states'][3].role == 'Patrol'
    assert bs['Alpha']['token_states'][3].position == 14
    assert bs['Alpha']['token_states'][4].role == 'Patrol'
    assert bs['Alpha']['token_states'][4].position == 14
    assert bs['Alpha']['token_states'][5].role == 'Patrol'
    assert bs['Alpha']['token_states'][5].position == 7
    assert bs['Alpha']['token_states'][6].role == 'Patrol'
    assert bs['Alpha']['token_states'][6].position == 7
    assert bs['Alpha']['token_states'][7].role == 'Patrol'
    assert bs['Alpha']['token_states'][7].position == 8
    assert bs['Alpha']['token_states'][8].role == 'Patrol'
    assert bs['Alpha']['token_states'][8].position == 8
    assert bs['Alpha']['token_states'][9].role == 'Patrol'
    assert bs['Alpha']['token_states'][9].position == 8
    assert bs['Alpha']['token_states'][10].role == 'Patrol'
    assert bs['Alpha']['token_states'][10].position == 9

    assert bs['Bravo']['token_states'][0].role == 'HVA'
    assert bs['Bravo']['token_states'][0].position == 11
    assert bs['Bravo']['token_states'][1].role == 'Patrol'
    assert bs['Bravo']['token_states'][1].position == 9
    assert bs['Bravo']['token_states'][2].role == 'Patrol'
    assert bs['Bravo']['token_states'][2].position == 10
    assert bs['Bravo']['token_states'][3].role == 'Patrol'
    assert bs['Bravo']['token_states'][3].position == 10
    assert bs['Bravo']['token_states'][4].role == 'Patrol'
    assert bs['Bravo']['token_states'][4].position == 10
    assert bs['Bravo']['token_states'][5].role == 'Patrol'
    assert bs['Bravo']['token_states'][5].position == 11
    assert bs['Bravo']['token_states'][6].role == 'Patrol'
    assert bs['Bravo']['token_states'][6].position == 11
    assert bs['Bravo']['token_states'][7].role == 'Patrol'
    assert bs['Bravo']['token_states'][7].position == 12
    assert bs['Bravo']['token_states'][8].role == 'Patrol'
    assert bs['Bravo']['token_states'][8].position == 12
    assert bs['Bravo']['token_states'][9].role == 'Patrol'
    assert bs['Bravo']['token_states'][9].position == 12
    assert bs['Bravo']['token_states'][10].role == 'Patrol'
    assert bs['Bravo']['token_states'][10].position == 13

@pytest.mark.parametrize(
    "move_seq,exp_pos", [
        (MOVE_SEQ_0,EXP_POS_0)
    ]
)
def test_KOTHGame_move_pieces(move_seq, exp_pos):
    game = koth.KOTHGame(
        max_ring=5, 
        min_ring=1, 
        geo_ring=4,
        init_board_pattern_p1=INIT_BOARD_PATTERN_1,
        init_board_pattern_p2=INIT_BOARD_PATTERN_1,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_1, init_pattern_beta=INIT_BOARD_PATTERN_1)
    assert game.game_state[U.P1][U.TOKEN_STATES][0].position == exp_pos[0][0]
    assert game.game_state[U.P2][U.TOKEN_STATES][0].position == exp_pos[0][1]
    for i, move_i in enumerate(move_seq):
        game.move_pieces(move_i)
        assert game.game_state[U.P1][U.TOKEN_STATES][0].position == exp_pos[i+1][0]
        assert game.game_state[U.P2][U.TOKEN_STATES][0].position == exp_pos[i+1][1]

# @settings(max_examples=8, verbosity=Verbosity.verbose)
@given( ordered_integer_triplet_st, game_init_pattern_st )
def deprecated_test_hypothesis_KOTHGame_get_observation(ring_def, init_pattern):
    max_ring = ring_def[2]
    min_ring = ring_def[0]
    geo_ring = ring_def[1]
    game = koth.KOTHGame(
        max_ring=max_ring, 
        min_ring=min_ring, 
        geo_ring=geo_ring,
        init_board_pattern_p1=init_pattern,
        init_board_pattern_p2=init_pattern,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=init_pattern, init_pattern_beta=init_pattern)
    na = game.n_tokens_alpha
    nb = game.n_tokens_beta
    p1_obs = game.get_observation('Alpha')
    p2_obs = game.get_observation('Bravo')
    assert np.all(p1_obs[0:na] == p2_obs[nb:])
    assert np.all(p1_obs[na:] == p2_obs[0:nb])

@given( game_state_st() )
def deprecated_test_hypothesis_KOTHGame_get_observation(game):
    na = game.n_tokens_alpha
    nb = game.n_tokens_beta
    p1_obs = game.get_observation('Alpha')
    p2_obs = game.get_observation('Bravo')
    assert np.all(p1_obs[0:na] == p2_obs[nb:])
    assert np.all(p1_obs[na:] == p2_obs[0:nb])

# @given( ordered_integer_triplet_st, game_init_pattern_st )
@given( game_state_st() )
def test_hypothesis_KOTHGame_token_catalog(game):
    '''ensure piece catelog and game state are synced 
    '''
    # max_ring = ring_def[2]
    # min_ring = ring_def[0]
    # geo_ring = ring_def[1]
    # game = koth.KOTHGame(max_ring=max_ring, min_ring=min_ring, geo_ring=geo_ring)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=init_pattern, init_pattern_beta=init_pattern)
    for _ in range(100):

        # select random piece
        player = np.random.choice([U.P1, U.P2])
        if player == U.P1:
            sat_i = np.random.randint(game.n_tokens_alpha)
        else:
            sat_i = np.random.randint(game.n_tokens_beta)
        piece_type = U.SEEKER if sat_i == 0 else U.BLUDGER
        piece_name = player + ':' + piece_type + ':' + str(sat_i)

        # modify piece state in game_state, check for consistency
        game.game_state[player][U.TOKEN_STATES][sat_i].position = np.random.randint(game.board_grid.n_sectors)
        game.game_state[player][U.TOKEN_STATES][sat_i].satellite.fuel = np.random.randint(101)
        assert game.token_catalog[piece_name].position == game.game_state[player][U.TOKEN_STATES][sat_i].position
        assert game.token_catalog[piece_name].satellite.fuel == game.game_state[player][U.TOKEN_STATES][sat_i].satellite.fuel

        # modify piece state in token_catalog, check for consistency
        game.token_catalog[piece_name].position = np.random.randint(game.board_grid.n_sectors)
        game.token_catalog[piece_name].satellite.fuel = np.random.randint(101)
        assert game.token_catalog[piece_name].position == game.game_state[player][U.TOKEN_STATES][sat_i].position
        assert game.token_catalog[piece_name].satellite.fuel == game.game_state[player][U.TOKEN_STATES][sat_i].satellite.fuel

def deprecated_test_KOTHGame_get_observation_0():
    game = koth.KOTHGame(max_ring=5, min_ring=1, geo_ring=4)
    game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
        game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_0, init_pattern_beta=INIT_BOARD_PATTERN_0)
    p1_obs = game.get_observation('Alpha')
    p2_obs = game.get_observation('Bravo')
    assert p1_obs.shape == (22, 63)
    assert p2_obs.shape == (22, 63)
    assert np.all(p1_obs[0:11] == p2_obs[11:])
    assert np.all(p1_obs[11:] == p2_obs[0:11])
    exp_p1_obs = np.zeros((22, 63))
    exp_p1_obs[0,15] = 100
    exp_p1_obs[1,29] = 100
    exp_p1_obs[2,30] = 100
    exp_p1_obs[3,30] = 100
    exp_p1_obs[4,30] = 100
    exp_p1_obs[5,15] = 100
    exp_p1_obs[6,15] = 100
    exp_p1_obs[7,16] = 100
    exp_p1_obs[8,16] = 100
    exp_p1_obs[9,16] = 100
    exp_p1_obs[10,17] = 100
    exp_p1_obs[11,23] = 100
    exp_p1_obs[12,21] = 100
    exp_p1_obs[13,22] = 100
    exp_p1_obs[14,22] = 100
    exp_p1_obs[15,22] = 100
    exp_p1_obs[16,23] = 100
    exp_p1_obs[17,23] = 100
    exp_p1_obs[18,24] = 100
    exp_p1_obs[19,24] = 100
    exp_p1_obs[20,24] = 100
    exp_p1_obs[21,25] = 100
    assert np.all(p1_obs == exp_p1_obs)

@pytest.mark.parametrize(
    "move_seq,exp_pos", [
        (MOVE_SEQ_0,EXP_POS_0)
    ]
)
def test_KOTHGame_apply_verbose_actions_move_0(move_seq, exp_pos):
    game = koth.KOTHGame(
        max_ring=5, 
        min_ring=1, 
        geo_ring=4,
        init_board_pattern_p1=INIT_BOARD_PATTERN_1,
        init_board_pattern_p2=INIT_BOARD_PATTERN_1,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_1, init_pattern_beta=INIT_BOARD_PATTERN_1)
    assert game.game_state[U.P1][U.TOKEN_STATES][0].position == exp_pos[0][0]
    assert game.game_state[U.P2][U.TOKEN_STATES][0].position == exp_pos[0][1]
    for i, move_i in enumerate(move_seq):
        game.apply_verbose_actions(move_i)
        assert game.game_state[U.P1][U.TOKEN_STATES][0].position == exp_pos[i+1][0]
        assert game.game_state[U.P2][U.TOKEN_STATES][0].position == exp_pos[i+1][1]

        # jump to next movement phase
        game.update_turn_phase(U.MOVEMENT)

@pytest.mark.parametrize(
    "engagements", [
        (ENGAGE_0)
    ]
)
def test_KOTHGame_apply_verbose_actions_engage_0(engagements):
    game = koth.KOTHGame(
        max_ring=1, 
        min_ring=1, 
        geo_ring=1,
        init_board_pattern_p1=INIT_BOARD_PATTERN_2,
        init_board_pattern_p2=INIT_BOARD_PATTERN_2,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_2, init_pattern_beta=INIT_BOARD_PATTERN_2)
    
    init_ammo_p1b1 = game.token_catalog[P1B1].satellite.ammo

    game.update_turn_phase(U.ENGAGEMENT)

    # apply engagements
    game.apply_verbose_actions(engagements)

    # check outcome is as expected
    assert game.token_catalog[P1B1].satellite.ammo == init_ammo_p1b1 - 1             # all ammo spent
    assert game.token_catalog[P2S0].satellite.fuel > DEFAULT_PARAMS_PARTIAL['min_fuel']  # satellite not destroyed because of guard
    assert game.token_catalog[P2B1].satellite.fuel == DEFAULT_PARAMS_PARTIAL['min_fuel'] # satellite destroyed because of guard
    assert game.token_catalog[P2B1].position == game.token_catalog[P2S0].position 

def test_KOTHGame_apply_verbose_actions_drift_0():
    game = koth.KOTHGame(
        max_ring=5, 
        min_ring=1, 
        geo_ring=4,
        init_board_pattern_p1=INIT_BOARD_PATTERN_0,
        init_board_pattern_p2=INIT_BOARD_PATTERN_0,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_0, init_pattern_beta=INIT_BOARD_PATTERN_0)
    game.update_turn_phase(U.DRIFT)

    # copy initial locations
    init_state = deepcopy(game.token_catalog)

    #Fuel points are calculated prior to the drift (which decrements fuel by 1)
    fuel_points_p1 = game.get_fuel_points(U.P1)
    fuel_points_p2 = game.get_fuel_points(U.P2)

    # apply drift
    game.apply_verbose_actions(None)

    # check outcome is as expected
    for token_name, token_state in game.token_catalog.items():
        assert token_state.position == game.board_grid.get_relative_azimuth_sector(init_state[token_name].position, 1)

    # check game score
    assert game.game_state[U.P1][U.SCORE] == DEFAULT_PARAMS_PARTIAL['in_goal_points'][U.P1] + fuel_points_p1
    assert game.game_state[U.P2][U.SCORE] == DEFAULT_PARAMS_PARTIAL['in_goal_points'][U.P2] + fuel_points_p2

@pytest.mark.parametrize(
    "engagements", [
        (ENGAGE_0)
    ]
)
def test_KOTHGame_resolve_engagements(engagements):
    game = koth.KOTHGame(
        max_ring=5, 
        min_ring=1, 
        geo_ring=4,
        init_board_pattern_p1=INIT_BOARD_PATTERN_2,
        init_board_pattern_p2=INIT_BOARD_PATTERN_2,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_2, init_pattern_beta=INIT_BOARD_PATTERN_2)
    game.resolve_engagements(engagements)

@pytest.mark.parametrize(
    "engagements", [
        (ENGAGE_0)
    ]
)
def test_get_illegal_verbose_actions_engage_0(engagements):
    game = koth.KOTHGame(
        max_ring=1, 
        min_ring=1, 
        geo_ring=1,
        init_board_pattern_p1=INIT_BOARD_PATTERN_2,
        init_board_pattern_p2=INIT_BOARD_PATTERN_2,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_2, init_pattern_beta=INIT_BOARD_PATTERN_2)
    game.game_state[U.TURN_PHASE] = U.ENGAGEMENT
    game.update_legal_verbose_actions()
    
    illegal_actions, alpha_illegal, beta_illegal = koth.get_illegal_verbose_actions(actions=engagements, legal_actions=game.game_state[U.LEGAL_ACTIONS])
    assert len(illegal_actions) == 0
    assert not alpha_illegal
    assert not beta_illegal

def test_KOTHGame_get_legal_verbose_actions_move_0():
    game = koth.KOTHGame(
        max_ring=5, 
        min_ring=1, 
        geo_ring=4,
        init_board_pattern_p1=INIT_BOARD_PATTERN_0,
        init_board_pattern_p2=INIT_BOARD_PATTERN_0,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    # game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_0, init_pattern_beta=INIT_BOARD_PATTERN_0)
    legal_actions = koth.get_legal_verbose_actions(
        turn_phase=game.game_state[U.TURN_PHASE],
        token_catalog=game.token_catalog,
        board_grid=game.board_grid,
        token_adjacency_graph=game.game_state[U.TOKEN_ADJACENCY],
        min_ring=game.inargs.min_ring,
        max_ring=game.inargs.max_ring)
    for pname, leg_act in legal_actions.items():
        assert all([U.MovementTuple(move) in leg_act for move in U.MOVEMENT_TYPES])

def check_legal_move_actions(game, legal_actions):
    for tok_name, leg_act in legal_actions.items():
        assert tok_name in game.token_catalog.keys()
        assert U.MovementTuple(U.NOOP) in leg_act
        if game.token_catalog[tok_name].satellite.fuel > 0: #This is a hard coded zero value. Should be self.inargs.min_fuel, but get_legal_verbose_actions does not have access to self.inargs...
            assert all([U.MovementTuple(move) in leg_act for move in [U.NOOP, U.PROGRADE, U.RETROGRADE]])
            if game.board_grid.sector_num2ring(game.token_catalog[tok_name].position) < game.inargs.max_ring:
                assert U.MovementTuple(U.RADIAL_OUT) in leg_act
            else:
                assert U.MovementTuple(U.RADIAL_OUT) not in leg_act
            if game.board_grid.sector_num2ring(game.token_catalog[tok_name].position) > game.inargs.min_ring:
                assert U.MovementTuple(U.RADIAL_IN) in leg_act
            else:
                assert U.MovementTuple(U.RADIAL_IN) not in leg_act

def check_legal_engage_actions(game, legal_actions):
    for tok_name, leg_act in legal_actions.items():
        assert tok_name in game.token_catalog.keys()
        if U.P1 in tok_name:
            plr_id = U.P1
        else:
            plr_id = U.P2
        assert U.EngagementTuple(U.NOOP, tok_name, None) in leg_act
        for oth_name in game.game_state[U.TOKEN_ADJACENCY].neighbors(tok_name):
            if game.token_catalog[tok_name].position == game.token_catalog[oth_name].position:
                same_sector = True
            else:
                same_sector = False
            if koth.is_same_player(tok_name, oth_name):
                #assert U.EngagementTuple(U.GUARD, oth_name, None) in leg_act #Guard is  only legal now if there is an adversary satellite adjacent to the HVA
                assert U.EngagementTuple(U.COLLIDE, oth_name, None) not in leg_act
                assert U.EngagementTuple(U.SHOOT, oth_name, None) not in leg_act
            else:
                assert U.EngagementTuple(U.GUARD, oth_name, None) not in leg_act
                if same_sector:
                    if game.token_catalog[tok_name].satellite.fuel >= game.inargs.fuel_usage[plr_id][U.IN_SEC][U.COLLIDE] and game.token_catalog[oth_name].satellite.fuel > 0 and U.SEEKER not in tok_name:
                        assert U.EngagementTuple(U.COLLIDE, oth_name, None) in leg_act
                    if game.token_catalog[tok_name].satellite.fuel >= game.inargs.fuel_usage[plr_id][U.IN_SEC][U.SHOOT] and game.token_catalog[tok_name].satellite.ammo >= 1 and game.token_catalog[oth_name].satellite.fuel > 0:
                        assert U.EngagementTuple(U.SHOOT, oth_name, None) in leg_act
                else:
                    if game.token_catalog[tok_name].satellite.fuel >= game.inargs.fuel_usage[plr_id][U.ADJ_SEC][U.COLLIDE] and game.token_catalog[oth_name].satellite.fuel > 0 and U.SEEKER not in tok_name:
                        assert U.EngagementTuple(U.COLLIDE, oth_name, None) in leg_act
                    if game.token_catalog[tok_name].satellite.fuel >= game.inargs.fuel_usage[plr_id][U.ADJ_SEC][U.SHOOT] and game.token_catalog[tok_name].satellite.ammo >= 1 and game.token_catalog[oth_name].satellite.fuel > 0:
                        assert U.EngagementTuple(U.SHOOT, oth_name, None) in leg_act



@given( ordered_integer_triplet_st, game_init_pattern_st )
@settings(deadline=None)
def test_hypothesis_get_legal_verbose_actions_move(ring_def, init_pattern):
    
    # setup game based on boad def and init token placement
    max_ring = ring_def[2]
    min_ring = ring_def[0]
    geo_ring = ring_def[1]
    game = koth.KOTHGame(
        max_ring=max_ring, 
        min_ring=min_ring, 
        geo_ring=geo_ring,
        init_board_pattern_p1=init_pattern,
        init_board_pattern_p2=init_pattern,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=init_pattern, init_pattern_beta=init_pattern)

    # get legal actions for this initial game state (i.e MOVEMENT)
    legal_actions = koth.get_legal_verbose_actions(
        turn_phase=game.game_state[U.TURN_PHASE],
        token_catalog=game.token_catalog,
        board_grid=game.board_grid,
        token_adjacency_graph=game.game_state[U.TOKEN_ADJACENCY],
        min_ring=game.inargs.min_ring,
        max_ring=game.inargs.max_ring)

    # check that expected movement actions are legal
    assert len(legal_actions) == len(game.token_catalog)
    check_legal_move_actions(game, legal_actions)
    

@given(game_state_st())
@settings(deadline=None)
def test_hypothesis_get_legal_verbose_actions(game):
    # get legal actions
    legal_actions = koth.get_legal_verbose_actions(
        turn_phase=game.game_state[U.TURN_PHASE],
        token_catalog=game.token_catalog,
        board_grid=game.board_grid,
        token_adjacency_graph=game.game_state[U.TOKEN_ADJACENCY],
        min_ring=game.inargs.min_ring,
        max_ring=game.inargs.max_ring)
    assert len(legal_actions) == len(game.token_catalog)
    if game.game_state[U.TURN_PHASE] == U.MOVEMENT:
        check_legal_move_actions(game, legal_actions)
    elif game.game_state[U.TURN_PHASE] == U.ENGAGEMENT:
        check_legal_engage_actions(game, legal_actions)

@given(game_state_st())
def test_hypothesis_KOTHGame_apply_verbose_actions_move(game):

    # ~~~ ARRANGE ~~~
    # jump to next movement phase
    game.update_turn_phase(U.MOVEMENT)

    # record pre-movement token states
    pre_tok_states = deepcopy(game.token_catalog)

    # get list of legal actions for each piece
    legal_actions = koth.get_legal_verbose_actions(
        turn_phase=game.game_state[U.TURN_PHASE],
        token_catalog=game.token_catalog,
        board_grid=game.board_grid,
        token_adjacency_graph=game.game_state[U.TOKEN_ADJACENCY],
        min_ring=game.inargs.min_ring,
        max_ring=game.inargs.max_ring)

    # generate random valid movement actions
    action_dict = dict()
    low_fuel_tokens = []
    for token_name, token_state in game.token_catalog.items():
        if U.P1 in token_name:
            plr_id = U.P1
        else:
            plr_id = U.P2
        # random valid movement
        action_ind = np.random.choice(len(legal_actions[token_name]))
        action = legal_actions[token_name][action_ind]

        # add to dict of all actions
        action_dict[token_name] = action

        # find low fuel tokens
        if token_state.satellite.fuel < game.inargs.fuel_usage[plr_id][action.action_type]:
            low_fuel_tokens.append(token_name)

    # ~~~ ACT ~~~
    # apply actions
    game.apply_verbose_actions(action_dict)

    # ~~~ ASSERT ~~~
    # assert low-fuel tokens don't move
    for lft in low_fuel_tokens:
        #assert game.token_catalog[lft].satellite.fuel == game.inargs.min_fuel #Removed this check as fuel is not decremented if there is not enough fuel to attempt the action in the first place
        assert game.token_catalog[lft].position == pre_tok_states[lft].position

    # assert non-low-fuel tokens move as expected
    other_tokens = [t for t in game.token_catalog.keys() if t not in low_fuel_tokens]
    for tok in other_tokens:
        act = action_dict[tok].action_type
        pre_sec = pre_tok_states[tok].position
        cur_sec = game.token_catalog[tok].position
        if act == U.PROGRADE:
            assert game.board_grid.get_prograde_sector(pre_sec) == cur_sec
        elif act == U.RETROGRADE:
            assert game.board_grid.get_retrograde_sector(pre_sec) == cur_sec
        elif act == U.RADIAL_IN:
            assert game.board_grid.get_radial_in_sector(pre_sec) == cur_sec
        elif act == U.RADIAL_OUT:
            assert game.board_grid.get_radial_out_sector(pre_sec) == cur_sec
        elif act == U.NOOP:
            assert pre_sec == cur_sec
        else:
            raise ValueError("Unrecognized action type: {}".format(act))


@given(game_state_st())
def test_hypothesis_KOTHGame_apply_fuel_constraints_move(game):

    action_dict = dict()
    low_fuel_tokens = []
    for token_name, token_state in game.token_catalog.items():
        if U.P1 in token_name:
            plr_id = U.P1
        else:
            plr_id = U.P2
        # random choose movement
        action = np.random.choice(U.MOVEMENT_TYPES)

        # create all movement tuple
        action_dict[token_name] = U.MovementTuple(action)

        # find low fuel tokens
        if token_state.satellite.fuel < game.inargs.fuel_usage[plr_id][action]:
            low_fuel_tokens.append(token_name)

    # apply fuel constraints
    fuel_constrained_action_dict = game.apply_fuel_constraints(action_dict)

    # check output
    for low_fuel_token in low_fuel_tokens:
        #assert game.token_catalog[low_fuel_token].satellite.fuel == game.inargs.min_fuel #if not enough fuel the action is not attempted and no fuel is decremented
        assert fuel_constrained_action_dict[low_fuel_token].action_type == U.NOOP

@given(game_state_st())
def test_hypothesis_KOTHGame_apply_fuel_constraints_engage(game):

    action_dict = dict()
    low_fuel_tokens = []
    for token_name, token_state in game.token_catalog.items():
        if U.P1 in token_name:
            plr_id = U.P1
        else:
            plr_id = U.P2
        # random choose engagement
        action = np.random.choice(U.ENGAGEMENT_TYPES+[U.NOOP])

        # randomly choose valid target 
        if action == U.NOOP:
            possible_targets = [token_name]
        elif action == U.GUARD:
            possible_targets = [tar for tar in 
            game.game_state[U.TOKEN_ADJACENCY].neighbors(token_name) if 
            koth.is_same_player(token_name, tar)]
        else:
            possible_targets = [tar for tar in 
                game.game_state[U.TOKEN_ADJACENCY].neighbors(token_name) if 
                not koth.is_same_player(token_name, tar)]

        # set to NOOP if no valid target for chosen action
        if len(possible_targets) == 0:
            action = U.NOOP
            target = token_name
        else:
            target = np.random.choice(possible_targets)

        # create engagement tuple
        action_dict[token_name] = U.EngagementTuple(action, target, None)

        # determine fuel usage of action
        fuel_usage = None
        if action == U.NOOP:
            fuel_usage = game.inargs.fuel_usage[plr_id][U.NOOP]
        else:
            if game.token_catalog[token_name].position == game.token_catalog[target].position:
                fuel_usage = game.inargs.fuel_usage[plr_id][U.IN_SEC][action]
            else:
                fuel_usage = game.inargs.fuel_usage[plr_id][U.ADJ_SEC][action]

        # find low fuel tokens
        if token_state.satellite.fuel < fuel_usage:
            low_fuel_tokens.append(token_name)

    # apply fuel constraints
    fuel_constrained_action_dict = game.apply_fuel_constraints(action_dict)

    # check output
    for lft in low_fuel_tokens:
        #assert game.token_catalog[lft].satellite.fuel == game.inargs.min_fuel #if not enough fuel the action is not attempted and no fuel is decremented
        assert fuel_constrained_action_dict[lft].action_type == U.NOOP


@given(game_state_st())
@settings(deadline=None)
def test_hypothesis_get_token_adjacency_graph(game):
    adj_graph = koth.get_token_adjacency_graph(game.board_grid, game.token_catalog)
    
    # check that number graph nodes equals number of tokens
    adj_graph_nodes = list(adj_graph.nodes())
    assert len(adj_graph_nodes) == len(game.token_catalog)

    # choose random token to manually inspect adjacency
    for _ in range(100):
        tok = np.random.choice(adj_graph_nodes)
        tok_sec = game.token_catalog[tok].position

        # check no self connections
        assert not adj_graph.has_edge(tok, tok)
        assert not game.game_state[U.TOKEN_ADJACENCY].has_edge(tok, tok)

        # find sectors adjacent to token position
        #adj_secs = list(game.board_grid.get_all_adjacent_sectors(tok_sec))
        #adj_secs.append(tok_sec)
        adj_secs = [
            tok_sec, 
            game.board_grid.get_relative_azimuth_sector(tok_sec,1),
            game.board_grid.get_relative_azimuth_sector(tok_sec,-1)]
        out_sec = game.board_grid.get_radial_out_sector(tok_sec)
        if out_sec is not None:
            adj_secs.append(out_sec)
            adj_secs.append(game.board_grid.get_relative_azimuth_sector(out_sec,1))
        in_sec = game.board_grid.get_radial_in_sector(tok_sec)
        if in_sec is not None:
            adj_secs.append(in_sec)

        # count pieces in adjacent sectors
        n_adj_tok = 0
        for tok2, tok2_state in game.token_catalog.items():
            debug_str = "DEBUG token1: {}-{}, token2: {}-{}, board: {},{},{}".format(
                tok, game.token_catalog[tok].position, 
                tok2, game.token_catalog[tok2].position,
                game.inargs.min_ring, game.inargs.max_ring, game.inargs.geo_ring)
            if tok == tok2:
                continue
            if tok2_state.position == tok_sec:
                assert adj_graph.has_edge(tok, tok2), debug_str
                assert game.game_state[U.TOKEN_ADJACENCY].has_edge(tok, tok2), debug_str
            if tok2_state.position in adj_secs:
                n_adj_tok += 1

        assert len(adj_graph.out_edges(tok)) == n_adj_tok


@given(game_state_st())
def test_hypothesis_KOTHGame_get_engagement_probability(game):

    # generate random engagements and test probabilities
    for _ in range(10):
        tokens = np.random.choice(list(game.token_catalog.keys()), size=2, replace=False)
        inst_tok = tokens[0]
        targ_tok = tokens[1]
        engagement = np.random.choice(U.ENGAGEMENT_TYPES+[U.NOOP])
        prob = game.get_engagement_probability(inst_tok, targ_tok, engagement)

        if U.P1 in inst_tok:
            plr_id = U.P1
        else:
            plr_id = U.P2

        # check in-sector probability
        # print("DEBUG: adj graph {}:{}".format(inst_tok, game.game_state[U.TOKEN_ADJACENCY].edges(inst_tok)))
        debug_str = "DEBUG inst_tok: {}-{}, tar_tok: {}-{}, adjacent:{}, engagement:{}, prob:{}".format(
            inst_tok, game.token_catalog[inst_tok].position, 
            targ_tok, game.token_catalog[targ_tok].position, 
            game.game_state[U.TOKEN_ADJACENCY].has_edge(inst_tok, targ_tok),
            engagement,
            prob)
        # print(debug_str)
        if engagement == U.NOOP:
            assert prob == game.inargs.engage_probs[plr_id][U.IN_SEC][U.NOOP], debug_str
        elif game.token_catalog[inst_tok].position == game.token_catalog[targ_tok].position:
            assert game.game_state[U.TOKEN_ADJACENCY].has_edge(inst_tok, targ_tok)
            assert prob == game.inargs.engage_probs[plr_id][U.IN_SEC][engagement],  debug_str
        elif game.game_state[U.TOKEN_ADJACENCY].has_edge(inst_tok, targ_tok):
            assert prob == game.inargs.engage_probs[plr_id][U.ADJ_SEC][engagement], debug_str
        else:
            assert prob == 0.0, debug_str

@given(game_state_st())
def test_hypothesis_KOTHGame_get_points(game):

    # get current points
    alpha_points, beta_points = game.get_points()

    # determine if alpha is on mission
    al_seek_sec = game.game_state[U.P1][U.TOKEN_STATES][0].position
    al_goal_sec = game.game_state[U.GOAL1]
    if al_seek_sec == al_goal_sec:
        assert alpha_points == game.inargs.in_goal_points[U.P1]
    elif al_seek_sec in game.board_grid.get_all_adjacent_sectors(al_goal_sec):
        assert alpha_points == game.inargs.adj_goal_points[U.P1]

    # determine if alpha is on mission
    bt_seek_sec = game.game_state[U.P2][U.TOKEN_STATES][0].position
    bt_goal_sec = game.game_state[U.GOAL2]
    if bt_seek_sec == bt_goal_sec:
        assert beta_points == game.inargs.in_goal_points[U.P2]
    elif bt_seek_sec in game.board_grid.get_all_adjacent_sectors(bt_goal_sec):
        assert beta_points == game.inargs.adj_goal_points[U.P2]

if __name__ == '__main__':
    # test_KOTHGame_initial_game_state()
    # test_KOTHGame_move_pieces(MOVE_SEQ_0,EXP_POS_0)
    # test_KOTHGame_get_observation_0()
    # test_KOTHGame_resolve_engagements(ENGAGE_0)
    # test_get_illegal_verbose_actions_engage_0(ENGAGE_0)
    pass
