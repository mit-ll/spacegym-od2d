# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# this is meant as a sandbox for running a complete koth game
# using a game server and agent clients 
# with random-yet-valid agents 

from multiprocessing import Value
import time
import zmq
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.game_server as GS
import orbit_defender2d.king_of_the_hill.default_game_parameters_old as DGP
from orbit_defender2d.king_of_the_hill import koth
from numpy.random import choice

DEFAULT_PARAMS_PARTIAL = {
    'init_fuel' : DGP.INIT_FUEL,
    'init_ammo' : DGP.INIT_AMMO,
    'min_fuel' : DGP.MIN_FUEL,
    'fuel_usage' : DGP.FUEL_USAGE,
    'engage_probs' : DGP.ENGAGE_PROBS,
    'illegal_action_score' : DGP.ILLEGAL_ACT_SCORE,
    'in_goal_points' : DGP.IN_GOAL_POINTS,
    'adj_goal_points' : DGP.ADJ_GOAL_POINTS,
    'fuel_points_factor' : DGP.FUEL_POINTS_FACTOR,
    'win_score' : DGP.WIN_SCORE,
    'max_turns' : DGP.MAX_TURNS}

INIT_BOARD_PATTERN_2 = [(0,1)]
INIT_BOARD_PATTERN_3 = [(-2,1), (-1,3), (0,2), (1,3), (2,1)] # (relative azim, number of pieces)

PORT_NUM = 5555
API_VER_NUM = 'v2021.07.06.0000'

ECHO_REQ_MSG_0 = {'context': 'echo', 'data': {'key0': 'value0'}}

def get_turn_init_req_msg():
    return {
        GS.API_VERSION:API_VER_NUM,
        GS.CONTEXT:GS.TURN_INIT
    }

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

def format_action_request(rep_game_state, req_actions):
    '''format verbose action dictionary into JSON request dictionary'''

    req_msg = dict()
    req_msg[GS.API_VERSION] = API_VER_NUM
    req_msg[GS.DATA] = dict()
    if rep_game_state[GS.TURN_PHASE] == U.MOVEMENT:
        req_msg[GS.CONTEXT] = GS.MOVE_PHASE
        req_msg[GS.DATA][GS.KIND] = GS.MOVE_PHASE_REQ
        req_msg[GS.DATA][GS.MOVEMENT_SELECTIONS] = req_actions
    elif rep_game_state[GS.TURN_PHASE] == U.ENGAGEMENT:
        req_msg[GS.CONTEXT] = GS.ENGAGE_PHASE
        req_msg[GS.DATA][GS.KIND] = GS.ENGAGE_PHASE_REQ
        req_msg[GS.DATA][GS.ENGAGEMENT_SELECTIONS] = req_actions
    elif rep_game_state[GS.TURN_PHASE] == U.DRIFT:
        req_msg[GS.CONTEXT] = GS.DRIFT_PHASE
        req_msg[GS.DATA][GS.KIND] = GS.DRIFT_PHASE_REQ
    else:
        raise ValueError("Unrecognized turn phase {}".format(rep_game_state[GS.TURN_PHASE]))

    return req_msg


def print_game_info(rep_game_state):
    print("STATES:")
    for tok in rep_game_state[GS.TOKEN_STATES]:
        print("   {:<16s}| position: {:<4d}| fuel: {:<8.1f} ".format(tok[GS.PIECE_ID], tok[GS.POSITION], tok[GS.FUEL]))
    print("alpha|beta score: {}|{}".format(rep_game_state[GS.SCORE_ALPHA],rep_game_state[GS.SCORE_BETA]))

def print_actions(req_actions):
    print("ACTIONS:")
    if req_actions is None:
        print("   None")
    else:
        for act in req_actions:
            print("   {:<15s} | {}".format(act[GS.PIECE_ID], act[GS.ACTION_TYPE]))

if __name__ == "__main__":

    # create game object
    game = koth.KOTHGame(
        max_ring=5, 
        min_ring=1, 
        geo_ring=4,
        init_board_pattern=INIT_BOARD_PATTERN_3,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_2, init_pattern_beta=INIT_BOARD_PATTERN_2)

    # create game server
    game_server = GS.GameServer(game, comm_configs={GS.TCP_PORT: PORT_NUM})

    # start game server object
    game_server.start()

    # get response and check for expected format
    
    i = 1
    while i < 6:
        print(i)
        i += 1
    rep_msg = ""
    while 0 < 1:
        print("Checking for message")
        time.sleep(5)
    print(rep_msg)      
            
    
    assert rep_msg[GS.API_VERSION] == API_VER_NUM
    assert rep_msg[GS.CONTEXT] == GS.TURN_INIT
    assert rep_msg[GS.DATA][GS.KIND] == GS.TURN_INIT_RESP

    

    # rep_game_state = rep_game_state = rep_msg[GS.DATA][GS.GAME_STATE]
    # assert rep_game_state[GS.TURN_PHASE] == U.MOVEMENT
    # assert_valid_game_state(rep_game_state)

    # print("\n<==== GAME INITILIZATION ====>")
    # print_game_info(rep_game_state=rep_game_state)

    # while not rep_msg[GS.DATA][GS.GAME_STATE][GS.GAME_DONE]:
    # # for iii in range(10):

        # print("\n<==== Turn: {} | Phase: {} ====>".format(rep_game_state[GS.TURN_NUMBER], rep_game_state[GS.TURN_PHASE]))

        # if rep_game_state[GS.TURN_PHASE] == U.DRIFT:
            # req_actions = None
            # req_msg = get_turn_init_req_msg()

        # else:
            # # select random valid action formatted as client request dictionary
            # req_actions = []
            # for tok in rep_game_state[GS.TOKEN_STATES]:
                # act = tok[GS.LEGAL_ACTIONS][choice(len(tok[GS.LEGAL_ACTIONS]))]
                # act[GS.PIECE_ID] = tok[GS.PIECE_ID]
                # req_actions.append(act)

            # # format action as client request
            # req_msg = format_action_request(rep_game_state, req_actions)

        # # send request to game server
        # print_actions(req_actions=req_actions)
        # players_client.send_json(req_msg)

        # # get response and check for validity
        # rep_msg = players_client.recv_json()
        # rep_game_state = rep_msg[GS.DATA][GS.GAME_STATE]
        # assert_valid_game_state(rep_game_state)
        
        # # print game state information
        # print_game_info(rep_game_state=rep_game_state)

    # cleanup
    # print("\n====GAME FINISHED====\n=====================\n")
    game_server.terminate()
    game_server.join()

    winner = None
    alpha_score = rep_game_state[GS.SCORE_ALPHA]
    beta_score = rep_game_state[GS.SCORE_BETA]
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'
    print("\n====GAME FINISHED====\nWinner: {}\nScore: {}|{}\n=====================\n".format(winner, alpha_score, beta_score))
