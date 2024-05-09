# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
import orbit_defender2d.king_of_the_hill.default_game_parameters as DGP
from hypothesis import given, settings, Verbosity
from hypothesis import strategies as st
from orbit_defender2d.king_of_the_hill import koth
from collections import OrderedDict

@st.composite
def game_state_nonterminal_st(draw):
    '''draw a valid positions, fuel, and ammo for a valid non-terminal game state'''

    # create game object with default params
    game = koth.KOTHGame(**PZE.GAME_PARAMS._asdict())

    min_sector = game.board_grid.sector_coord2num(game.inargs.min_ring, 0)
    max_sector = game.board_grid.n_sectors-1

    # generate list of valid, non-terminal fuel amounts
    fuels_alpha = draw(st.lists(
        elements=st.floats(
            min_value=DGP.MIN_FUEL, 
            max_value=max(DGP.INIT_FUEL[U.P1][U.SEEKER], DGP.INIT_FUEL[U.P1][U.BLUDGER]), 
            allow_infinity=False, 
            allow_nan=False, 
            exclude_min=True),
        min_size=game.n_tokens_alpha, max_size=game.n_tokens_alpha))
    fuels_beta = draw(st.lists(
        elements=st.floats(
            min_value=DGP.MIN_FUEL, 
            max_value=max(DGP.INIT_FUEL[U.P2][U.SEEKER], DGP.INIT_FUEL[U.P2][U.BLUDGER]), 
            allow_infinity=False, 
            allow_nan=False, 
            exclude_min=True),
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

    # generate valid non-terminal turn phase, turn count, hill locations, and  scores
    game.game_state[U.GOAL1] = draw(st.integers(min_value=min_sector, max_value=max_sector))
    game.game_state[U.GOAL2] = draw(st.integers(min_value=min_sector, max_value=max_sector))
    game.game_state[U.P1][U.SCORE] = draw(st.floats(
        min_value=0.0, 
        max_value=DGP.WIN_SCORE[U.P1], 
        allow_infinity=False, 
        allow_nan=False, 
        exclude_max=True))
    game.game_state[U.P2][U.SCORE] = draw(st.floats(
        min_value=0.0, 
        max_value=DGP.WIN_SCORE[U.P2], 
        allow_infinity=False, 
        allow_nan=False, 
        exclude_max=True))
    game.game_state[U.TURN_COUNT] = draw(st.integers(min_value=0, max_value=DGP.MAX_TURNS-1))
    turn_phase = draw(st.sampled_from(U.TURN_PHASE_LIST))
    game.update_turn_phase(turn_phase)

    # return new token game state
    return game

def test_KOTHActionSpaces_init():
    '''initialize KOTHActionSpaces without error'''
    # Arrange: create game and initialize KOTHActionSpaces without error
    game = koth.KOTHGame(**PZE.GAME_PARAMS._asdict())
    act_space_info = PZE.KOTHActionSpaces(game=game)

    # Act: None, test is just to initialize KOTHActionSpaces without error

    # Assert: None, test is just to initialize KOTHActionSpaces without error

def test_KOTHObservationSpaces_init():
    '''initialize KOTHObservationSpaces without error'''
    # Arrange: create game and initialize KOTHObservationSpaces without error
    game = koth.KOTHGame(**PZE.GAME_PARAMS._asdict())
    act_space_info = PZE.KOTHObservationSpaces(game=game)

    # Act: None, test is just to initialize KOTHObservationSpaces without error

    # Assert: None, test is just to initialize KOTHObservationSpaces without error

def test_parallel_env_init():
    '''create parallel_env instance without error'''
    # Arrange: create parallel_env instance without error
    parallel_env = PZE.parallel_env()

    # Act: none
    # Assert: none

@given(game_state_nonterminal_st())
def test_hypothesis_parallel_env_game_state_nonterminal(game):
    '''create parallel_env with arbitrary non-terminal game state without error'''
    # Arrange: create parallel_env
    parallel_env = PZE.parallel_env()

    # Act: replace game with arbitrary game state
    parallel_env.kothgame = game

    # Assert: check game is non-terminal
    assert not parallel_env.kothgame.game_state[U.GAME_DONE] 

@given(game_state_nonterminal_st())
@settings(deadline=None)
def test_hypothesis_parallel_env_encode_token_observation(game):
    '''test encode_token_observation function for runtime errors'''
    # ~~~ Arrange ~~~
    # create parallel_env
    parallel_env = PZE.parallel_env()

    # ~~~ Act ~~~
    # replace game with arbitrary game state
    parallel_env.kothgame = game
    
    # ~~~ Assert ~~~
    # get observation for randomly selected token and randomly
    # selected player to ensure no runtime errors
    # check for expected sizing
    for trial_i in range(64):
        # select random token
        tok_name = np.random.choice(list(parallel_env.kothgame.token_catalog.keys()))
        tok_state = parallel_env.kothgame.token_catalog[tok_name]
        # plr, _, _ = koth.parse_token_id(tok_name)

        # select random player
        plr = np.random.choice([U.P1, U.P2])

        # get observation of token from player perspective
        flat_obs, tuple_obs = parallel_env.encode_token_observation(plr, tok_name)

        # check sizing
        assert flat_obs.shape == (PZE.N_BITS_OBS_PER_TOKEN,)
        # assert flat_obs.dtype == 'float64'
        assert parallel_env.obs_space_info.flat_per_token.contains(flat_obs)
        assert len(tuple_obs) == 5
        
        # check decoding
        ind1 = 0
        ind2 = PZE.N_BITS_OBS_OWN_PIECE
        decoded_own_piece = flat_obs[ind1:ind2]
        assert decoded_own_piece == (tok_name.split(U.TOKEN_DELIMITER)[0] == plr)
        assert len(tuple_obs[0]) == 1
        assert tuple_obs[0][0] == decoded_own_piece
        
        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_ROLE
        decoded_role = flat_obs[ind1:ind2]  # one-hot
        decoded_role = np.nonzero(decoded_role)[0]    # discrete
        assert decoded_role.shape == (1,)   # ensure it was in fact a one-hot
        decoded_role = decoded_role[0] 
        assert decoded_role == U.PIECE_ROLES.index(tok_state.role)
        assert tuple_obs[1] == decoded_role

        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_POSITION
        decoded_position = flat_obs[ind1:ind2]  # one-hot
        decoded_position = np.nonzero(decoded_position)[0]  # discrete
        assert decoded_position.shape == (1,)   # ensure one-hot encoding
        decoded_position = decoded_position[0]
        assert decoded_position == tok_state.position
        assert tuple_obs[2] == decoded_position

        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_FUEL
        decoded_fuel = flat_obs[ind1:ind2]  # binary
        decoded_fuel = U.bitlist2int(decoded_fuel)  # integer
        assert decoded_fuel == int(tok_state.satellite.fuel)
        assert U.bitlist2int(tuple_obs[3]) == decoded_fuel

        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_AMMO
        assert ind2 == len(flat_obs)  # ensure vector length
        decoded_ammo = flat_obs[ind1:]  # binary
        decoded_ammo = U.bitlist2int(decoded_ammo)
        assert decoded_ammo == tok_state.satellite.ammo
        assert U.bitlist2int(tuple_obs[4]) == decoded_ammo

@given(game_state_nonterminal_st())
def test_hypothesis_parallel_env_encode_scoreboard_observation(game):
    '''test encode_token_observation function for runtime errors'''
    # ~~~ ARRANGE ~~~
    # create parallel_env
    parallel_env = PZE.parallel_env()

    # ~~~ ACT ~~~
    # replace game with arbitrary game state and collect both player's obs
    parallel_env.kothgame = game
    p1_flat_obs, p1_dict_obs = parallel_env.encode_scoreboard_observation(U.P1)
    p2_flat_obs, p2_dict_obs = parallel_env.encode_scoreboard_observation(U.P2)

    # ~~~ ASSERT ~~~
    # step through flat observation verifying values

    assert parallel_env.obs_space_info.flat_scoreboard.contains(p1_flat_obs)
    assert parallel_env.obs_space_info.scoreboard.contains(p1_dict_obs)
    assert parallel_env.obs_space_info.flat_scoreboard.contains(p2_flat_obs)
    assert parallel_env.obs_space_info.scoreboard.contains(p1_dict_obs)

    # turn phase check
    assert p1_dict_obs['turn_phase'] == p2_dict_obs['turn_phase']
    ind1 = 0
    ind2 = PZE.N_BITS_OBS_TURN_PHASE
    p1_decoded_turn_phase = p1_flat_obs[ind1:ind2] # one-hot
    p2_decoded_turn_phase = p2_flat_obs[ind1:ind2] # one-hot
    assert (p1_decoded_turn_phase == p2_decoded_turn_phase).all()   # ensure p1 identical to p2 obs
    p1_decoded_turn_phase = np.nonzero(p1_decoded_turn_phase)[0] # discrete
    assert p1_decoded_turn_phase.shape == (1,) # ensure one-hot
    p1_decoded_turn_phase = p1_decoded_turn_phase[0]
    assert p1_decoded_turn_phase == U.TURN_PHASE_LIST.index(game.game_state[U.TURN_PHASE])
    assert p1_decoded_turn_phase == p1_dict_obs['turn_phase']

    assert (p1_dict_obs['turn_count'] == p2_dict_obs['turn_count']).all()
    ind1 = ind2
    ind2 += PZE.N_BITS_OBS_TURN_COUNT
    p1_decoded_turn_count = p1_flat_obs[ind1:ind2]  # binary
    p2_decoded_turn_count = p2_flat_obs[ind1:ind2]  # binary
    assert (p1_decoded_turn_count == p2_decoded_turn_count).all() # ensure p1 identical to p2 obs
    p1_decoded_turn_count = U.bitlist2int(p1_decoded_turn_count) # integer
    assert p1_decoded_turn_count == game.game_state[U.TURN_COUNT]
    assert p1_decoded_turn_count == U.bitlist2int(p1_dict_obs['turn_count'])

    ind1 = ind2
    ind2 += PZE.N_BITS_OBS_SCORE
    p1_decoded_own_score = p1_flat_obs[ind1:ind2]   # binary
    p2_decoded_own_score = p2_flat_obs[ind1:ind2]   # binary
    assert p1_decoded_own_score[0] == p1_dict_obs['own_score'][0] == 1 # check ownership bool
    assert p2_decoded_own_score[0] == p2_dict_obs['own_score'][0] == 1 # check ownership bool
    p1_decoded_own_score = p1_decoded_own_score[1:] # remove bool own score
    p2_decoded_own_score = p2_decoded_own_score[1:] # remove bool own score
    assert (p1_decoded_own_score == p1_dict_obs['own_score'][1]).all()  # check match with dict obs
    assert (p2_decoded_own_score == p2_dict_obs['own_score'][1]).all()  # check match with dict obs
    p1_decoded_own_score_is_neg = p1_decoded_own_score[0] 
    p2_decoded_own_score_is_neg = p2_decoded_own_score[0] 
    p1_decoded_own_score = U.bitlist2int(p1_decoded_own_score[1:])  # integer
    p2_decoded_own_score = U.bitlist2int(p2_decoded_own_score[1:])  # integer
    if p1_decoded_own_score_is_neg:
        p1_decoded_own_score = -p1_decoded_own_score
    if p2_decoded_own_score_is_neg:
        p2_decoded_own_score = -p2_decoded_own_score
    assert p1_decoded_own_score == int(game.game_state[U.P1][U.SCORE])
    assert p2_decoded_own_score == int(game.game_state[U.P2][U.SCORE])

    ind1 = ind2
    ind2 += PZE.N_BITS_OBS_SCORE
    p1_decoded_opp_score = p1_flat_obs[ind1:ind2]   # binary
    p2_decoded_opp_score = p2_flat_obs[ind1:ind2]   # binary
    assert p1_decoded_opp_score[0] == p1_dict_obs['opponent_score'][0] == 0 # check ownership bool
    assert p2_decoded_opp_score[0] == p2_dict_obs['opponent_score'][0] == 0 # check ownership bool
    p1_decoded_opp_score = p1_decoded_opp_score[1:] # truncate ownership bool
    p2_decoded_opp_score = p2_decoded_opp_score[1:] # truncate ownership bool
    assert (p1_decoded_opp_score == p1_dict_obs['opponent_score'][1]).all() # check dict obs matching
    assert (p2_decoded_opp_score == p2_dict_obs['opponent_score'][1]).all() # check dict obs matching
    p1_decoded_opp_score_is_neg = p1_decoded_opp_score[0]
    p2_decoded_opp_score_is_neg = p2_decoded_opp_score[0]
    p1_decoded_opp_score = U.bitlist2int(p1_decoded_opp_score[1:])  # integer
    p2_decoded_opp_score = U.bitlist2int(p2_decoded_opp_score[1:])  # integer
    if p1_decoded_opp_score_is_neg:
        p1_decoded_opp_score = -p1_decoded_opp_score
    if p2_decoded_opp_score_is_neg:
        p2_decoded_opp_score = -p2_decoded_opp_score
    assert p1_decoded_opp_score == int(game.game_state[U.P2][U.SCORE])
    assert p2_decoded_opp_score == int(game.game_state[U.P1][U.SCORE])
    assert p1_decoded_opp_score == p2_decoded_own_score
    assert p2_decoded_opp_score == p1_decoded_own_score

    ind1 = ind2
    ind2 += PZE.N_BITS_OBS_HILL
    p1_decoded_own_hill = p1_flat_obs[ind1:ind2]    # one-hot
    p2_decoded_own_hill = p2_flat_obs[ind1:ind2]    # one-hot
    assert p1_decoded_own_hill[0] == p1_dict_obs['own_hill'][0] == 1    # check ownership bool
    assert p2_decoded_own_hill[0] == p2_dict_obs['own_hill'][0] == 1    # check ownership bool
    p1_decoded_own_hill = np.nonzero(p1_decoded_own_hill[1:])[0]   # truncate ownership bool and convert discrete
    p2_decoded_own_hill = np.nonzero(p2_decoded_own_hill[1:])[0]   # truncate ownership bool and convert discrete
    assert p1_decoded_own_hill.shape == (1,)    # ensure was valid one-hot
    assert p2_decoded_own_hill.shape == (1,)    # ensure was valid one-hot
    p1_decoded_own_hill = p1_decoded_own_hill[0]
    p2_decoded_own_hill = p2_decoded_own_hill[0]
    assert p1_decoded_own_hill == p1_dict_obs['own_hill'][1] ==  game.game_state[U.GOAL1]
    assert p2_decoded_own_hill == p2_dict_obs['own_hill'][1] == game.game_state[U.GOAL2]

    ind1 = ind2
    ind2 += PZE.N_BITS_OBS_HILL
    assert ind2 == len(p1_flat_obs) == len(p2_flat_obs) # check total vector length
    p1_decoded_opp_hill = p1_flat_obs[ind1:]    # one-hot
    p2_decoded_opp_hill = p2_flat_obs[ind1:]    # one-hot
    assert p1_decoded_opp_hill[0] == p1_dict_obs['opponent_hill'][0] == 0    # check ownership bool
    assert p2_decoded_opp_hill[0] == p2_dict_obs['opponent_hill'][0] == 0    # check ownership bool
    p1_decoded_opp_hill = np.nonzero(p1_decoded_opp_hill[1:])[0]   # truncate ownership bool and convert discrete
    p2_decoded_opp_hill = np.nonzero(p2_decoded_opp_hill[1:])[0]   # truncate ownership bool and convert discrete
    assert p1_decoded_opp_hill.shape == (1,)    # ensure was valid one-hot
    assert p2_decoded_opp_hill.shape == (1,)    # ensure was valid one-hot
    p1_decoded_opp_hill = p1_decoded_opp_hill[0]
    p2_decoded_opp_hill = p2_decoded_opp_hill[0]
    assert p1_decoded_opp_hill == p1_dict_obs['opponent_hill'][1] == game.game_state[U.GOAL2]
    assert p2_decoded_opp_hill == p2_dict_obs['opponent_hill'][1] == game.game_state[U.GOAL1]
    assert p1_decoded_opp_hill == p2_decoded_own_hill
    assert p2_decoded_opp_hill == p1_decoded_own_hill

@given(game_state_nonterminal_st())
@settings(deadline=None)
def test_hypothesis_parallel_env_encode_player_observation(game):
    '''test encode_token_observation function for runtime errors'''
    # ~~~ ARRANGE ~~~
    # create parallel_env
    penv = PZE.parallel_env()

    # ~~~ ACT ~~~
    # replace game with arbitrary game state and collect both player's obs
    penv.kothgame = game
    penv.reset()
    p1_flat_obs, p1_dict_obs = penv.encode_player_observation(U.P1)
    p2_flat_obs, p2_dict_obs = penv.encode_player_observation(U.P2)

    # ~~~ ASSERT ~~~

    assert penv.obs_space_info.flat_per_player.contains(p1_flat_obs)
    assert penv.obs_space_info.per_player.contains(p1_dict_obs)
    assert penv.obs_space_info.flat_per_player.contains(p2_flat_obs)
    assert penv.obs_space_info.per_player.contains(p1_dict_obs)

    # step through flat observation verifying values
    # turn phase check
    assert p1_dict_obs['scoreboard']['turn_phase'] == p2_dict_obs['scoreboard']['turn_phase']
    ind1 = 0
    ind2 = PZE.N_BITS_OBS_TURN_PHASE
    p1_decoded_turn_phase = p1_flat_obs[ind1:ind2] # one-hot
    p2_decoded_turn_phase = p2_flat_obs[ind1:ind2] # one-hot
    assert (p1_decoded_turn_phase == p2_decoded_turn_phase).all()   # ensure p1 identical to p2 obs
    p1_decoded_turn_phase = np.nonzero(p1_decoded_turn_phase)[0] # discrete
    assert p1_decoded_turn_phase.shape == (1,) # ensure one-hot
    p1_decoded_turn_phase = p1_decoded_turn_phase[0]
    assert p1_decoded_turn_phase == U.TURN_PHASE_LIST.index(game.game_state[U.TURN_PHASE])
    assert p1_decoded_turn_phase == p1_dict_obs['scoreboard']['turn_phase']

    # turn count check
    assert (p1_dict_obs['scoreboard']['turn_count'] == p2_dict_obs['scoreboard']['turn_count']).all()
    ind1 = ind2
    ind2 += PZE.N_BITS_OBS_TURN_COUNT
    p1_decoded_turn_count = p1_flat_obs[ind1:ind2]  # binary
    p2_decoded_turn_count = p2_flat_obs[ind1:ind2]  # binary
    assert (p1_decoded_turn_count == p2_decoded_turn_count).all() # ensure p1 identical to p2 obs
    p1_decoded_turn_count = U.bitlist2int(p1_decoded_turn_count) # integer
    assert p1_decoded_turn_count == game.game_state[U.TURN_COUNT]
    assert p1_decoded_turn_count == U.bitlist2int(p1_dict_obs['scoreboard']['turn_count'])

    # check own-score
    ind1 = ind2
    ind2 += PZE.N_BITS_OBS_SCORE
    p1_decoded_own_score = p1_flat_obs[ind1:ind2]   # binary
    p2_decoded_own_score = p2_flat_obs[ind1:ind2]   # binary
    assert p1_decoded_own_score[0] == p1_dict_obs['scoreboard']['own_score'][0] == 1 # check ownership bool
    assert p2_decoded_own_score[0] == p2_dict_obs['scoreboard']['own_score'][0] == 1 # check ownership bool
    p1_decoded_own_score = p1_decoded_own_score[1:] # remove ownership bool
    p2_decoded_own_score = p2_decoded_own_score[1:] # remove ownership bool
    assert (p1_decoded_own_score == p1_dict_obs['scoreboard']['own_score'][1]).all()  # check match with dict obs
    assert (p2_decoded_own_score == p2_dict_obs['scoreboard']['own_score'][1]).all()  # check match with dict obs
    p1_decoded_own_score = U.bitlist2int(p1_decoded_own_score)  # integer
    p2_decoded_own_score = U.bitlist2int(p2_decoded_own_score)  # integer
    assert p1_decoded_own_score == int(game.game_state[U.P1][U.SCORE])
    assert p2_decoded_own_score == int(game.game_state[U.P2][U.SCORE])

    # check opponent score
    ind1 = ind2
    ind2 += PZE.N_BITS_OBS_SCORE
    p1_decoded_opp_score = p1_flat_obs[ind1:ind2]   # binary
    p2_decoded_opp_score = p2_flat_obs[ind1:ind2]   # binary
    assert p1_decoded_opp_score[0] == p1_dict_obs['scoreboard']['opponent_score'][0] == 0 # check ownership bool
    assert p2_decoded_opp_score[0] == p2_dict_obs['scoreboard']['opponent_score'][0] == 0 # check ownership bool
    p1_decoded_opp_score = p1_decoded_opp_score[1:] # truncate ownership bool
    p2_decoded_opp_score = p2_decoded_opp_score[1:] # truncate ownership bool
    assert (p1_decoded_opp_score == p1_dict_obs['scoreboard']['opponent_score'][1]).all() # check dict obs matching
    assert (p2_decoded_opp_score == p2_dict_obs['scoreboard']['opponent_score'][1]).all() # check dict obs matching
    p1_decoded_opp_score = U.bitlist2int(p1_decoded_opp_score)  # integer
    p2_decoded_opp_score = U.bitlist2int(p2_decoded_opp_score)  # integer
    assert p1_decoded_opp_score == int(game.game_state[U.P2][U.SCORE])
    assert p2_decoded_opp_score == int(game.game_state[U.P1][U.SCORE])
    assert p1_decoded_opp_score == p2_decoded_own_score
    assert p2_decoded_opp_score == p1_decoded_own_score

    # check own hill
    ind1 = ind2
    ind2 += PZE.N_BITS_OBS_HILL
    p1_decoded_own_hill = p1_flat_obs[ind1:ind2]    # one-hot
    p2_decoded_own_hill = p2_flat_obs[ind1:ind2]    # one-hot
    assert p1_decoded_own_hill[0] == p1_dict_obs['scoreboard']['own_hill'][0] == 1    # check ownership bool
    assert p2_decoded_own_hill[0] == p2_dict_obs['scoreboard']['own_hill'][0] == 1    # check ownership bool
    p1_decoded_own_hill = np.nonzero(p1_decoded_own_hill[1:])[0]   # truncate ownership bool and convert discrete
    p2_decoded_own_hill = np.nonzero(p2_decoded_own_hill[1:])[0]   # truncate ownership bool and convert discrete
    assert p1_decoded_own_hill.shape == (1,)    # ensure was valid one-hot
    assert p2_decoded_own_hill.shape == (1,)    # ensure was valid one-hot
    p1_decoded_own_hill = p1_decoded_own_hill[0]
    p2_decoded_own_hill = p2_decoded_own_hill[0]
    assert p1_decoded_own_hill == p1_dict_obs['scoreboard']['own_hill'][1] ==  game.game_state[U.GOAL1]
    assert p2_decoded_own_hill == p2_dict_obs['scoreboard']['own_hill'][1] == game.game_state[U.GOAL2]

    # check opponent hill
    ind1 = ind2
    ind2 += PZE.N_BITS_OBS_HILL
    p1_decoded_opp_hill = p1_flat_obs[ind1:ind2]    # one-hot
    p2_decoded_opp_hill = p2_flat_obs[ind1:ind2]    # one-hot
    assert p1_decoded_opp_hill[0] == p1_dict_obs['scoreboard']['opponent_hill'][0] == 0    # check ownership bool
    assert p2_decoded_opp_hill[0] == p2_dict_obs['scoreboard']['opponent_hill'][0] == 0    # check ownership bool
    p1_decoded_opp_hill = np.nonzero(p1_decoded_opp_hill[1:])[0]   # truncate ownership bool and convert discrete
    p2_decoded_opp_hill = np.nonzero(p2_decoded_opp_hill[1:])[0]   # truncate ownership bool and convert discrete
    assert p1_decoded_opp_hill.shape == (1,)    # ensure was valid one-hot
    assert p2_decoded_opp_hill.shape == (1,)    # ensure was valid one-hot
    p1_decoded_opp_hill = p1_decoded_opp_hill[0]
    p2_decoded_opp_hill = p2_decoded_opp_hill[0]
    assert p1_decoded_opp_hill == p1_dict_obs['scoreboard']['opponent_hill'][1] == game.game_state[U.GOAL2]
    assert p2_decoded_opp_hill == p2_dict_obs['scoreboard']['opponent_hill'][1] == game.game_state[U.GOAL1]
    assert p1_decoded_opp_hill == p2_decoded_own_hill
    assert p2_decoded_opp_hill == p1_decoded_own_hill

    # check length of flattened scoreboard obs
    assert ind2 == PZE.N_BITS_OBS_SCOREBOARD

    # seperate own player tokens in token_catalog
    p1_token_states = OrderedDict([
        (tok_n, tok_s) for tok_n, tok_s in 
        penv.kothgame.token_catalog.items() if koth.parse_token_id(tok_n)[0] == U.P1])
    p2_token_states = OrderedDict([
        (tok_n, tok_s) for tok_n, tok_s in 
        penv.kothgame.token_catalog.items() if koth.parse_token_id(tok_n)[0] == U.P2])

    # check own-tokens of p1
    check_tokens_per_player(
        player_id = U.P1,
        token_flat_obs = p1_flat_obs[PZE.N_BITS_OBS_SCOREBOARD:PZE.N_BITS_OBS_SCOREBOARD+PZE.N_BITS_OBS_TOKENS_PER_PLAYER],
        token_tuple_obs = p1_dict_obs['own_tokens'],
        token_state_info = p1_token_states,
        which_tokens='own_tokens')

    # check own-tokens of p2
    check_tokens_per_player(
        player_id = U.P2,
        token_flat_obs = p2_flat_obs[PZE.N_BITS_OBS_SCOREBOARD:PZE.N_BITS_OBS_SCOREBOARD+PZE.N_BITS_OBS_TOKENS_PER_PLAYER],
        token_tuple_obs = p2_dict_obs['own_tokens'],
        token_state_info = p2_token_states,
        which_tokens='own_tokens')

    # check opponent-tokens of p1
    check_tokens_per_player(
        player_id = U.P1,
        token_flat_obs = p1_flat_obs[PZE.N_BITS_OBS_SCOREBOARD+PZE.N_BITS_OBS_TOKENS_PER_PLAYER:],
        token_tuple_obs = p1_dict_obs['opponent_tokens'],
        token_state_info = p2_token_states,
        which_tokens='opponent_tokens')

    # check opponent-tokens of p1
    check_tokens_per_player(
        player_id = U.P2,
        token_flat_obs = p2_flat_obs[PZE.N_BITS_OBS_SCOREBOARD+PZE.N_BITS_OBS_TOKENS_PER_PLAYER:],
        token_tuple_obs = p2_dict_obs['opponent_tokens'],
        token_state_info = p1_token_states,
        which_tokens='opponent_tokens')
        
        
def check_tokens_per_player(player_id, token_flat_obs, token_tuple_obs, token_state_info, which_tokens):
    '''check that a player's token observations match the underlying token state info'''

    if which_tokens == 'own_tokens':
        bool_ownership = True
    elif which_tokens == 'opponent_tokens':
        bool_ownership = False
    else:
        raise ValueError('Unexpected which_tokens value: {}'.format(which_tokens))

    ind1 = 0
    ind2 = 0
    for tok_name, tok_state in token_state_info.items():

        # parse player name, token role, and token number from token ground state
        tok_player, tok_role, tok_num = koth.parse_token_id(tok_name)
        tok_num = int(tok_num)
        tok_tup_obs = token_tuple_obs[tok_num]
        if bool_ownership:
            assert tok_player == player_id
        else:
            assert tok_player != player_id

        # decode and check ownership
        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_OWN_PIECE
        decoded_own_piece = token_flat_obs[ind1:ind2]
        assert decoded_own_piece == bool_ownership
        assert len(tok_tup_obs[0]) == 1
        assert tok_tup_obs[0] == decoded_own_piece

        # decode and check role
        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_ROLE
        decoded_role = token_flat_obs[ind1:ind2]  # one-hot
        decoded_role = np.nonzero(decoded_role)[0]    # discrete array
        assert decoded_role.shape == (1,)   # ensure it was in fact a one-hot
        decoded_role = decoded_role[0] # discrete value
        assert decoded_role == U.PIECE_ROLES.index(tok_state.role)
        assert tok_tup_obs[1] == decoded_role

        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_POSITION
        decoded_position = token_flat_obs[ind1:ind2]  # one-hot
        decoded_position = np.nonzero(decoded_position)[0]  # discrete array
        assert decoded_position.shape == (1,)   # ensure one-hot encoding
        decoded_position = decoded_position[0]  # discrete value
        assert decoded_position == tok_state.position
        assert tok_tup_obs[2] == decoded_position

        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_FUEL
        decoded_fuel = token_flat_obs[ind1:ind2]  # binary
        decoded_fuel = U.bitlist2int(decoded_fuel)  # integer
        assert decoded_fuel == int(tok_state.satellite.fuel)
        assert U.bitlist2int(tok_tup_obs[3]) == decoded_fuel

        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_AMMO
        decoded_ammo = token_flat_obs[ind1:ind2]  # binary
        decoded_ammo = U.bitlist2int(decoded_ammo)  # integer
        assert decoded_ammo == tok_state.satellite.ammo
        assert U.bitlist2int(tok_tup_obs[4]) == decoded_ammo

def check_all_encoded_observations(observations, game):
    '''check all encoded observations match game state
    
    Args:
        observaitons : dict
            key is agent (player) id
            value is flat, gym-encoded observation for respective agent
    '''

    for agent_id, agent_obs_dict in observations.items():

        # seperate observation from action_mask
        agent_obs = agent_obs_dict['observation']

        # get opponent id
        opp_id = [aid for aid in observations.keys() if aid != agent_id][0]

        # check decoded scoreboard
        ind1 = 0
        ind2 = PZE.N_BITS_OBS_TURN_PHASE
        decoded_turn_phase = agent_obs[ind1:ind2] # one-hot
        decoded_turn_phase = np.nonzero(decoded_turn_phase)[0] # discrete
        assert decoded_turn_phase.shape == (1,) # ensure one-hot
        decoded_turn_phase = decoded_turn_phase[0]
        assert decoded_turn_phase == U.TURN_PHASE_LIST.index(game.game_state[U.TURN_PHASE])

        # turn count check
        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_TURN_COUNT
        decoded_turn_count = agent_obs[ind1:ind2]  # binary
        decoded_turn_count = U.bitlist2int(decoded_turn_count) # integer
        assert decoded_turn_count == game.game_state[U.TURN_COUNT]

        # check own-score
        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_SCORE
        decoded_own_score = agent_obs[ind1:ind2]   # binary
        assert decoded_own_score[0] == 1 # check ownership bool
        decoded_own_score = decoded_own_score[1:] # remove ownership bool
        decoded_own_score_is_neg = decoded_own_score[0] # get sign
        decoded_own_score = U.bitlist2int(decoded_own_score[1:])  # unsigned integer
        if decoded_own_score_is_neg:
            decoded_own_score = -decoded_own_score
        assert decoded_own_score == int(game.game_state[agent_id][U.SCORE])

        # check opponent score
        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_SCORE
        decoded_opp_score = agent_obs[ind1:ind2]   # binary
        assert decoded_opp_score[0] == 0 # check ownership bool
        decoded_opp_score = decoded_opp_score[1:] # truncate ownership bool
        decoded_opp_score_is_neg = decoded_opp_score[0]  # get sign
        decoded_opp_score = U.bitlist2int(decoded_opp_score[1:])  # unsigned integer
        if decoded_opp_score_is_neg:
            decoded_opp_score = -decoded_opp_score
        assert decoded_opp_score == int(game.game_state[opp_id][U.SCORE])

        # check own hill
        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_HILL
        decoded_own_hill = agent_obs[ind1:ind2]    # one-hot
        assert decoded_own_hill[0] == 1    # check ownership bool
        decoded_own_hill = np.nonzero(decoded_own_hill[1:])[0]   # truncate ownership bool and convert discrete
        assert decoded_own_hill.shape == (1,)    # ensure was valid one-hot
        decoded_own_hill = decoded_own_hill[0]
        if agent_id == U.P1:
            assert decoded_own_hill == game.game_state[U.GOAL1]
        else:
            assert decoded_own_hill == game.game_state[U.GOAL2]

        # check opponent hill
        ind1 = ind2
        ind2 += PZE.N_BITS_OBS_HILL
        decoded_opp_hill = agent_obs[ind1:ind2]    # one-hot
        assert decoded_opp_hill[0] == 0    # check ownership bool
        decoded_opp_hill = np.nonzero(decoded_opp_hill[1:])[0]   # truncate ownership bool and convert discrete
        assert decoded_opp_hill.shape == (1,)    # ensure was valid one-hot
        decoded_opp_hill = decoded_opp_hill[0]
        if agent_id == U.P1:
            assert decoded_opp_hill == game.game_state[U.GOAL2]
        else:
            assert decoded_opp_hill == game.game_state[U.GOAL1]

        for tok_list_count in range(2):

            # own tokens are observed first
            bool_ownership = 1 if tok_list_count == 0 else 0

            for tok_num, encoded_tok_state in enumerate(agent_obs[
                PZE.N_BITS_OBS_SCOREBOARD + PZE.N_BITS_OBS_TOKENS_PER_PLAYER * tok_list_count:
                PZE.N_BITS_OBS_SCOREBOARD + PZE.N_BITS_OBS_TOKENS_PER_PLAYER * (tok_list_count + 1):
                PZE.N_BITS_OBS_PER_TOKEN]):

                # get token_id and token state from number and player id and game state
                if bool_ownership:
                    tok_id = game.get_token_id(player_id=agent_id, token_num=tok_num)
                else:
                    tok_id = game.get_token_id(player_id=opp_id, token_num=tok_num)
                tok_state = game.token_catalog[tok_id]

                # decode and check ownership
                ind1 = ind2
                ind2 += PZE.N_BITS_OBS_OWN_PIECE
                decoded_own_piece = agent_obs[ind1:ind2]
                assert decoded_own_piece == bool_ownership

                # decode and check role
                ind1 = ind2
                ind2 += PZE.N_BITS_OBS_ROLE
                decoded_role = agent_obs[ind1:ind2]  # one-hot
                decoded_role = np.nonzero(decoded_role)[0]    # discrete array
                assert decoded_role.shape == (1,)   # ensure it was in fact a one-hot
                decoded_role = decoded_role[0] # discrete value
                assert decoded_role == U.PIECE_ROLES.index(tok_state.role)

                ind1 = ind2
                ind2 += PZE.N_BITS_OBS_POSITION
                decoded_position = agent_obs[ind1:ind2]  # one-hot
                decoded_position = np.nonzero(decoded_position)[0]  # discrete array
                assert decoded_position.shape == (1,)   # ensure one-hot encoding
                decoded_position = decoded_position[0]  # discrete value
                assert decoded_position == tok_state.position

                ind1 = ind2
                ind2 += PZE.N_BITS_OBS_FUEL
                decoded_fuel = agent_obs[ind1:ind2]  # binary
                decoded_fuel = U.bitlist2int(decoded_fuel)  # integer
                assert decoded_fuel == int(tok_state.satellite.fuel)

                ind1 = ind2
                ind2 += PZE.N_BITS_OBS_AMMO
                decoded_ammo = agent_obs[ind1:ind2]  # binary
                decoded_ammo = U.bitlist2int(decoded_ammo)  # integer
                assert decoded_ammo == tok_state.satellite.ammo


@given(game_state_nonterminal_st())
@settings(deadline=None)
def test_hypothesis_parallel_env_decode_encode_token_action(game):
    '''decode then re-encode random gym action to check consistency'''

    # skip drift phase testing
    if game.game_state[U.TURN_PHASE] == U.DRIFT:
        return

    # ~~~ ARRANGE ~~~
    # create parallel_env
    penv = PZE.parallel_env()

    # ~~~ ACT ~~~
    # replace game with arbitrary game state and collect both player's obs
    penv.kothgame = game
    penv.reset()

    for trial_i in range(64):
        # draw random token and one-hot action
        rand_tok = np.random.choice(list(penv.kothgame.token_catalog.keys()))
        rand_flat_act = np.zeros(PZE.N_BITS_ACT_PER_TOKEN, dtype=penv.act_space_info.flat_per_token.dtype)
        rand_disc_act = np.random.randint(PZE.N_BITS_ACT_PER_TOKEN)
        rand_flat_act[rand_disc_act] = 1

        # decode random action into verbose action tuple
        decoded_disc_action = penv.decode_discrete_token_action(token_id=rand_tok, token_act=rand_disc_act)
        decoded_flat_action = penv.decode_flat_token_action(token_id=rand_tok, token_act=rand_flat_act)
        assert decoded_disc_action == decoded_flat_action

        # encode random action into 1D gym vector
        encoded_flat_action = penv.encode_flat_token_action(token_act=decoded_flat_action)
        encoded_disc_action = penv.encode_discrete_token_action(token_act=decoded_disc_action)

        # ~~~ ASSERT ~~~
        # check that encoded action matches original random action
        assert np.allclose(rand_flat_act, encoded_flat_action)
        assert rand_disc_act == encoded_disc_action

@given(game_state_nonterminal_st())
@settings(deadline=None)
def test_hypothesis_parallel_env_encode_decode_player_action(game):
    '''encode then decode random verbose player action to check consistency'''

    # skip drift phase testing
    if game.game_state[U.TURN_PHASE] == U.DRIFT:
        return

    # ~~~ ARRANGE ~~~
    # create parallel_env
    penv = PZE.parallel_env()

    # ~~~ ACT ~~~
    # replace game with arbitrary game state and collect both player's obs
    penv.kothgame = game
    penv.reset()

    for trial_i in range(64):

        # get random action
        rand_acts = penv.kothgame.get_random_valid_actions()
        p1_rand_acts = {tok_id: tok_act for tok_id, tok_act in rand_acts.items() if koth.parse_token_id(tok_id)[0]==U.P1}
        p2_rand_acts = {tok_id: tok_act for tok_id, tok_act in rand_acts.items() if koth.parse_token_id(tok_id)[0]==U.P2}
        assert len(p1_rand_acts) + len(p2_rand_acts) == len(rand_acts)

        # encode actions
        p1_flat_encoded_acts = penv.encode_flat_player_action(p1_rand_acts)
        p2_flat_encoded_acts = penv.encode_flat_player_action(p2_rand_acts)
        p1_disc_encoded_acts = penv.encode_discrete_player_action(p1_rand_acts)
        p2_disc_encoded_acts = penv.encode_discrete_player_action(p2_rand_acts)

        # decoded actions
        p1_decoded_acts_from_flat = penv.decode_flat_player_action(U.P1, p1_flat_encoded_acts)
        p2_decoded_acts_from_flat = penv.decode_flat_player_action(U.P2, p2_flat_encoded_acts)
        p1_decoded_acts_from_disc = penv.decode_discrete_player_action(U.P1, p1_disc_encoded_acts)
        p2_decoded_acts_from_disc = penv.decode_discrete_player_action(U.P2, p2_disc_encoded_acts)

        # ~~~ ASSERT ~~~
        # check decoded acts match original rand acts
        assert p1_decoded_acts_from_flat == p1_decoded_acts_from_disc == p1_rand_acts
        assert p2_decoded_acts_from_flat == p2_decoded_acts_from_disc == p2_rand_acts

@given(game_state_nonterminal_st())
@settings(deadline=None)
def test_hypothesis_parallel_env_decode_encode_player_action(game):
    '''decode then re-encode random gym player action to check consistency'''

    # skip drift phase testing
    if game.game_state[U.TURN_PHASE] == U.DRIFT:
        return

    # ~~~ ARRANGE ~~~
    # create parallel_env
    penv = PZE.parallel_env()

    # ~~~ ACT ~~~
    # replace game with arbitrary game state and collect both player's obs
    penv.kothgame = game
    penv.reset()

    for trial_i in range(64):
        # draw random token and one-hot action
        plr_id = np.random.choice(penv.kothgame.player_names)
        rand_disc_act = np.zeros(penv.n_tokens_per_player, dtype=penv.act_space_info.flat_per_player.dtype)
        rand_flat_act = np.zeros(( penv.n_tokens_per_player, PZE.N_BITS_ACT_PER_TOKEN), 
            dtype=penv.act_space_info.flat_per_player.dtype)
        for tok_num in range (penv.n_tokens_per_player):
            rand_disc_act[tok_num] = np.random.randint(PZE.N_BITS_ACT_PER_TOKEN)
            rand_flat_act[tok_num, rand_disc_act[tok_num]] = 1
        rand_flat_act = rand_flat_act.reshape((PZE.N_BITS_ACT_PER_PLAYER,))
        rand_disc_act = tuple(rand_disc_act)

        decoded_flat_act = penv.decode_flat_player_action(player_id=plr_id, player_act=rand_flat_act)
        decoded_disc_act = penv.decode_discrete_player_action(player_id=plr_id, player_act=rand_disc_act)

        encoded_flat_act = penv.encode_flat_player_action(decoded_flat_act)
        encoded_disc_act = penv.encode_discrete_player_action(decoded_disc_act)

        # ~~~ ASSERT ~~~
        # check that encoded action matches original random action
        assert np.allclose(encoded_flat_act, rand_flat_act)
        assert np.allclose(encoded_disc_act, rand_disc_act)
        assert decoded_disc_act == decoded_flat_act

@given(game_state_nonterminal_st())
@settings(deadline=None)
def test_hypothesis_parallel_env_encode_decode_all_actions(game):
    '''encode then decode random verbose actions to check consistency'''

    # skip drift phase testing
    if game.game_state[U.TURN_PHASE] == U.DRIFT:
        return

    # ~~~ ARRANGE ~~~
    # create parallel_env
    penv = PZE.parallel_env()

    # ~~~ ACT ~~~
    # replace game with arbitrary game state and collect both player's obs
    penv.kothgame = game
    penv.reset()

    for trial_i in range(64):

        # get random valid actions
        rand_acts = penv.kothgame.get_random_valid_actions()
        
        # encode random actions for all players
        encoded_flat_acts = penv.encode_all_flat_actions(actions=rand_acts)
        encoded_disc_acts = penv.encode_all_discrete_actions(actions=rand_acts)

        # decoded actions
        decoded_flat_acts = penv.decode_all_flat_actions(actions=encoded_flat_acts)
        decoded_disc_acts = penv.decode_all_discrete_actions(actions=encoded_disc_acts)

        # ~~~ ASSERT ~~~
        # check decoded acts match original rand acts
        assert decoded_flat_acts == decoded_disc_acts == rand_acts

@given(game_state_nonterminal_st())
# @settings(deadline=None)
def test_hypothesis_parallel_env_encode_decode_legal_action_mask(game):
    '''test decode_token_action function for runtime errors'''

    # skip drift phase testing
    if game.game_state[U.TURN_PHASE] == U.DRIFT:
        return

    # ~~~ ARRANGE ~~~
    # create parallel_env
    penv = PZE.parallel_env()

    # ~~~ ACT ~~~
    # replace game with arbitrary game state
    penv.kothgame = game
    penv.reset()

    # collect both player's legal action masks
    p1_encoded_legal_acts_mask = penv.encode_legal_action_mask(U.P1)
    p1_decoded_legal_act_mask = penv.decode_legal_action_mask(U.P1, p1_encoded_legal_acts_mask)
    p2_encoded_legal_acts_mask = penv.encode_legal_action_mask(U.P2)
    p2_decoded_legal_act_mask = penv.decode_legal_action_mask(U.P2, p2_encoded_legal_acts_mask)

    # merge decoded action masks
    decoded_legal_act_mask = {**p1_decoded_legal_act_mask, **p2_decoded_legal_act_mask}

    # ~~~ ASSERT ~~~
    # check that decoding of encoded action mask matches game state
    # Note: direct dictionary comparison not possible because 
    # action decoding process assign probability but game_state legal
    # actions have None set for probability 
    # for tok_id, tok_legacts in decoded_legal_act_mask.items():
    #     assert koth.is_legal_verbose_action(token=tok_id, action=to)
    legal_acts = penv.kothgame.game_state[U.LEGAL_ACTIONS]
    for tok_id, tok_legacts in legal_acts.items():
        decoded_tok_legacts = decoded_legal_act_mask[tok_id]
        assert len(decoded_tok_legacts) == len(tok_legacts)
        assert len(decoded_tok_legacts) == len(set(decoded_tok_legacts))
        for dec_legact in decoded_tok_legacts:
            assert koth.is_legal_verbose_action(token=tok_id, action=dec_legact, legal_actions=legal_acts), (
                "\ntok: {}\nact: {}\nlegal_acts {}".format(tok_id, dec_legact, legal_acts[tok_id]))

@given(game_state_nonterminal_st())
@settings(deadline=None)
def test_hypothesis_parallel_env_random_valid_step(game):
    '''check that stepping random valid action does not cause error'''

    # ~~~ ARRANGE ~~~
    # create parallel_env
    penv = PZE.parallel_env()

    # ~~~ ACT ~~~
    # replace game with arbitrary game state
    penv.kothgame = game
    penv.reset() 

    # prev_score = {ag_id: penv.kothgame.game_state[ag_id][U.SCORE] for ag_id in penv.agents}

    for trial_i in range(64):

        # get random valid verbose actions
        # ver_act = penv.kothgame.get_random_valid_actions()
        
        # encode random valid actions into gym space for all players
        # act = penv.encode_all_actions(actions=ver_act)
        act = penv.encode_random_valid_discrete_actions()

        # step game
        obs, rew, dones, info = penv.step(actions=act)

        # ~~~ ASSERT ~~~
        # check observations
        check_all_encoded_observations(obs, penv.kothgame)

        # check dones
        assert dones[U.P1] == dones[U.P2]

        # check rewards
        assert np.isclose(rew[U.P1], -rew[U.P2])
        if not dones[U.P1]:
            assert np.isclose(rew[U.P1], 0.0)
        else:
            assert np.isclose(rew[U.P1],
                penv.kothgame.game_state[U.P1][U.SCORE] - 
                penv.kothgame.game_state[U.P2][U.SCORE])
            break

@given(game_state_nonterminal_st())
@settings(deadline=None)
def test_hypothesis_parallel_env_split_random_valid_step(game):
    '''check that stepping random and random-valid actions terminates '''

    # ~~~ ARRANGE ~~~
    # create parallel_env
    penv = PZE.parallel_env()

    # ~~~ ACT ~~~
    # replace game with arbitrary game state
    penv.kothgame = game
    penv.reset() 

    prev_score = {ag_id: penv.kothgame.game_state[ag_id][U.SCORE] for ag_id in penv.agents}


    # encode random valid actions into gym space for all players
    # act = penv.encode_all_actions(actions=ver_act)
    acts = penv.encode_random_valid_discrete_actions()

    # choose a player for truly random actions
    rand_plr = np.random.choice([U.P1, U.P2])
    acts[rand_plr] = penv.action_spaces[rand_plr].sample()
    # rand_plr_acts_flat = penv.act_space_info.flat_per_player.sample()
    # decoded_rand_plr_acts = penv.decode_flat_player_action(rand_plr, rand_plr_acts_flat)
    # acts[rand_plr] = penv.encode_discrete_player_action(decoded_rand_plr_acts)

    # check that random action produces an invalid action
    # assert any(np.array(list(penv.decode_flat_player_action(rand_plr, rand_plr_acts_flat).values())) == U.INVALID_ACTION)

    # step game
    obs, rew, dones, info = penv.step(actions=acts)

    # ~~~ ASSERT ~~~
    # check observations
    check_all_encoded_observations(obs, penv.kothgame)

    # check dones
    assert dones[U.P1] == dones[U.P2]
    # check rewards
    assert np.isclose(rew[U.P1], -rew[U.P2])
    assert np.isclose(rew[U.P1],
        penv.kothgame.game_state[U.P1][U.SCORE] - 
        penv.kothgame.game_state[U.P2][U.SCORE])

def test_parallel_env_step():
    '''check that stepping random action from init game statedoes not cause error'''

    # ~~~ ARRANGE ~~~
    # create parallel_env
    penv = PZE.parallel_env()

    # ~~~ ACT ~~~
    penv.reset() 

    # prev_score = {ag_id: penv.kothgame.game_state[ag_id][U.SCORE] for ag_id in penv.agents}

    for trial_i in range(128):

        # get random valid verbose actions
        # ver_act = penv.kothgame.get_random_valid_actions()
        
        # encode random valid actions into gym space for all players
        # act = penv.encode_all_actions(actions=ver_act)
        act = penv.encode_random_valid_discrete_actions()

        # step game
        obs, rew, dones, info = penv.step(actions=act)

        # ~~~ ASSERT ~~~
        # check observations
        check_all_encoded_observations(obs, penv.kothgame)

        # check dones
        assert dones[U.P1] == dones[U.P2]

        # check rewards
        assert np.isclose(rew[U.P1], -rew[U.P2])
        if not dones[U.P1]:
            assert np.isclose(rew[U.P1], 0.0)
        else:
            assert np.isclose(rew[U.P1],
                penv.kothgame.game_state[U.P1][U.SCORE] - 
                penv.kothgame.game_state[U.P2][U.SCORE])
            break


if __name__ == "__main__":
    # test_KOTHActionSpaces_init()
    # test_KOTHObservationSpaces_init()
    test_parallel_env_step()
