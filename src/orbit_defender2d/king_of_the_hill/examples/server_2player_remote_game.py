import zmq
import time
import threading
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.default_game_parameters as DGP
import orbit_defender2d.king_of_the_hill.game_server as GS
from copy import deepcopy
from orbit_defender2d.king_of_the_hill import koth
from orbit_defender2d.king_of_the_hill.examples.server_utils import \
    assert_valid_game_state, print_game_info
from numpy.random import choice, rand, shuffle
from time import sleep

ROUTER_PORT_NUM = 5555
PUB_PORT_NUM = 5556

API_VER_NUM_2P = "v2021.11.18.0000.2p"

ECHO_REQ_MSG_0 = {'context': 'echo', 'data': {'key0': 'value0'}}

# Game Parameters
GAME_PARAMS = koth.KOTHGameInputArgs(
    max_ring=DGP.MAX_RING,
    min_ring=DGP.MIN_RING,
    geo_ring=DGP.GEO_RING,
    init_board_pattern_p1=DGP.INIT_BOARD_PATTERN_P1,
    init_board_pattern_p2=DGP.INIT_BOARD_PATTERN_P2,
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
    fuel_points_factor_bludger=DGP.FUEL_POINTS_FACTOR_BLUDGER,
    )


def run_server_2player_remote_game():

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

    # create and connect players' client
    print("Waiting on players to connec to router port {}".format(ROUTER_PORT_NUM))
    # alpha_client = context.socket(zmq.REQ)
    # alpha_client.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))

    while 1 > 0:
        # for iii in range(10):
        print("I think I am still running and handling stuff...")
        time.sleep(5)

    # cleanup
    print("Terminating server...")
    game_server.terminate()
    game_server.join()


if __name__ == "__main__":
    run_server_2player_remote_game()
