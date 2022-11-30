# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.game_server as GS

def assert_valid_game_state(game_state):
    '''check response from game server gives valid game state
    '''
    
    # turn counter
    assert isinstance(game_state[GS.TURN_NUMBER], int)
    assert game_state[GS.TURN_NUMBER] >= 0

    # turn phase
    assert game_state[GS.TURN_PHASE] in U.TURN_PHASE_LIST

    # game done
    assert isinstance(game_state[GS.GAME_DONE], bool)

    # goal positions
    assert isinstance(game_state[GS.GOAL_ALPHA], int)
    # assert 0 < game_state[GS.GOAL_ALPHA] <= game_board.n_sectors
    assert isinstance(game_state[GS.GOAL_BETA], int)
    # assert 0 < game_state[GS.GOAL_BETA] <= game_board.n_sectors

def print_game_info(game_state):
    print("STATES:")
    for tok in game_state[GS.TOKEN_STATES]:
        print("   {:<16s}| position: {:<4d}| fuel: {:<8.1f} ".format(tok[GS.PIECE_ID], tok[GS.POSITION], tok[GS.FUEL]))
    print("alpha|beta score: {}|{}".format(game_state[GS.SCORE_ALPHA],game_state[GS.SCORE_BETA]))
