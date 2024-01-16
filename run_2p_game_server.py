#Need this to start the game server and register the alpah client. Another script will register the beta client.
#I could do this as a stand alone server program, but would have to make sure to send reset requests from teh clients when they join and not sure how to 
#Register new player does a reset on the game, so that could work...

import numpy as np
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
import orbit_defender2d.king_of_the_hill.default_game_parameters as DGP
#import orbit_defender2d.king_of_the_hill.default_game_parameters_small as DGP
from orbit_defender2d.king_of_the_hill import koth
from orbit_defender2d.king_of_the_hill import game_server as GS

from time import sleep

# Game Parameters
GAME_PARAMS = koth.KOTHGameInputArgs(
    max_ring = DGP.MAX_RING,
    min_ring = DGP.MIN_RING,
    geo_ring = DGP.GEO_RING,
    init_board_pattern = DGP.INIT_BOARD_PATTERN,
    init_fuel = DGP.INIT_FUEL,
    init_ammo = DGP.INIT_AMMO,
    min_fuel = DGP.MIN_FUEL,
    fuel_usage = DGP.FUEL_USAGE,
    engage_probs = DGP.ENGAGE_PROBS,
    illegal_action_score = DGP.ILLEGAL_ACT_SCORE,
    in_goal_points = DGP.IN_GOAL_POINTS,
    adj_goal_points = DGP.ADJ_GOAL_POINTS,
    fuel_points_factor = DGP.FUEL_POINTS_FACTOR,
    win_score = DGP.WIN_SCORE,
    max_turns = DGP.MAX_TURNS)


ROUTER_PORT_NUM = 5555
PUB_PORT_NUM = 5556

API_VER_NUM_2P = "v2022.07.26.0000.2p"

ECHO_REQ_MSG_0 = {'context': 'echo', 'data': {'key0': 'value0'}}

def run_game():
    # create game object
    game = koth.KOTHGame(**GAME_PARAMS._asdict())

    # create game server
    comm_configs = {
        GS.ROUTER_PORT: ROUTER_PORT_NUM,
        GS.PUB_PORT: PUB_PORT_NUM
    }
    game_server = GS.TwoPlayerGameServer(game=game, comm_configs=comm_configs)

    # start game server object
    game_server.start()

    # remove local game objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game

    while True:
        sleep(30)
        print("Game server still running")

if __name__ == "__main__":
    
    run_game()
