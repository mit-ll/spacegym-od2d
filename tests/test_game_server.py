# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import zmq
import pytest
import uuid
from copy import deepcopy

import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.game_server as GS
import orbit_defender2d.king_of_the_hill.default_game_parameters as DGP
from orbit_defender2d.king_of_the_hill import koth
from orbit_defender2d.king_of_the_hill.game_server import \
    SingleUserGameServer, TwoPlayerGameServer, \
    TCP_PORT, ROUTER_PORT, PUB_PORT

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
    'max_turns' : DGP.MAX_TURNS}

DEFAULT_PARAMS = {
    'max_ring' : DGP.MAX_RING,
    'min_ring' : DGP.MIN_RING,
    'geo_ring' : DGP.GEO_RING,
    'init_board_pattern' : DGP.INIT_BOARD_PATTERN,
    **DEFAULT_PARAMS_PARTIAL
}

PORT_NUM = 5555
ROUTER_PORT_NUM = 5555
PUB_PORT_NUM = 5556
API_VER_NUM = 'v2021.11.18.0000.1p'
API_VER_NUM_2P = "v2022.07.26.0000.2p"


INIT_BOARD_PATTERN_2 = [(0,1)] 

ECHO_REQ_MSG_0 = {'context': 'echo', 'data': {'key0': 'value0'}}

GAME_RESET_REQ_MSG = {
    'apiVersion': API_VER_NUM,
    'context': 'gameReset'
}

DRIFT_REQ_MSG = {
    'apiVersion': API_VER_NUM,
    'context': 'driftPhase'
}

ENG_REQ_MSG_0 = {
    'apiVersion': API_VER_NUM,
    'context': 'engagementPhase',
    'data': {
        'kind': 'engagementPhaseRequest',
        'engagementSelections': [
            {'pieceID': 'alpha:seeker:0', 'actionType':'noop', 'targetID':'alpha:seeker:0'},
            {'pieceID': 'alpha:bludger:1', 'actionType':'shoot', 'targetID':'beta:seeker:0'},
            {'pieceID': 'beta:seeker:0', 'actionType':'noop', 'targetID':'beta:seeker:0'},
            {'pieceID': 'beta:bludger:1', 'actionType':'guard', 'targetID':'beta:seeker:0'},
        ]
    }}

MOV_REQ_MSG_0 = {
    'apiVersion': API_VER_NUM,
    'context': 'movementPhase',
    'data': {
        'kind': 'movementPhaseRequest',
        'movementSelections': [
            {'pieceID': 'alpha:seeker:0', 'actionType':'noop'},
            {'pieceID': 'alpha:bludger:1', 'actionType':'prograde'},
            {'pieceID': 'beta:seeker:0', 'actionType':'noop'},
            {'pieceID': 'beta:bludger:1', 'actionType':'retrograde'}
        ]
    }}

class ErrorCatchingSocket(object):
    def __init__(self, context, sock_type):
        self._context = context
        self._sock = context.socket(sock_type)

    def connect(self, *args, **kwargs):
        return self._sock.connect(*args, **kwargs)

    def send_json(self, *args, **kwargs):
        return self._sock.send_json(*args, **kwargs)
    
    def recv_json(self, *args, **kwargs):
        for i in range(100):
            try:
                rep = self._sock.recv_json(zmq.NOBLOCK)
                break
            except:
                import time; time.sleep(0.01)
        else:
            raise zmq.ZMQError('Got no answer.')

        return rep

@pytest.fixture
def single_user_game_server_fixture(request):
    """Creates single-user game server that terminates on request for test(s)."""
    def _game_server(game):
        pu_srv = SingleUserGameServer(game, {TCP_PORT: PORT_NUM})

        # Terminate the process when done with the test
        def terminate():
            pu_srv.terminate()
            pu_srv.join()

        request.addfinalizer(terminate)

        return pu_srv
    return _game_server

@pytest.fixture
def two_player_game_server_fixture(request):
    """Creates two-player game server that terminates on request for test(s)."""
    def _game_server(game):
        pu_srv = TwoPlayerGameServer(
            game=game, 
            comm_configs={
                ROUTER_PORT: ROUTER_PORT_NUM,
                PUB_PORT: PUB_PORT_NUM
            }
        )

        # Terminate the process when done with the test
        def terminate():
            pu_srv.terminate()
            pu_srv.join()

        request.addfinalizer(terminate)

        return pu_srv
    return _game_server

def test_TwoPlayerGameServer_echo_request(two_player_game_server_fixture):
    """Tests if the echo request responds properly for two-player server
    
    Ref:
        http://lists.idyll.org/pipermail/testing-in-python/2011-October/004507.html
    """

    # ~~~ ARRANGE ~~~
    # start the python-unity server
    game_server = two_player_game_server_fixture(None)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game_server

    # create a client and connect to matching port
    context = zmq.Context()
    req_sock = ErrorCatchingSocket(context, zmq.REQ)
    req_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))

    # ~~~ ACT ~~~
    # create simple dictionary and send as json to be echoed
    req_msg = ECHO_REQ_MSG_0
    req_sock.send_json(req_msg)

    # get response and check for matching
    rep_msg = req_sock.recv_json()

    # ~~~ ASSERT ~~~
    assert rep_msg == ECHO_REQ_MSG_0

def test_TwoPlayerGameServer_handle_invalid_api(two_player_game_server_fixture):
    """ test that invalid api version returns error message
    """

    # ~~~ ARRANGE ~~~
    # start the python-unity server
    game_server = two_player_game_server_fixture(None)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game_server

    # ~~~ ACT ~~~
    # create a client and connect to matching port
    context = zmq.Context()
    req_sock = ErrorCatchingSocket(context, zmq.REQ)
    req_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))

    # send request and get response
    req_msg = deepcopy(GAME_RESET_REQ_MSG)
    req_msg['apiVersion'] = 'v0000.00.00.2p'    # an invalid api version
    req_sock.send_json(req_msg)
    rep_msg = req_sock.recv_json()

    # ~~~ ASSERT ~~~
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'gameReset'
    # just check an error message exists, not what is contained within the message
    assert 'error' in rep_msg.keys()
    assert 'message' in rep_msg['error'].keys()

def test_TwoPlayerGameServer_handle_invalid_context(two_player_game_server_fixture):
    """ test that out-of-context requests return error message
    """

    # ~~~ ARRANGE ~~~
    # create game object with default params
    game = koth.KOTHGame(**DEFAULT_PARAMS)
    game.update_turn_phase(U.DRIFT)

    # start the python-unity server
    game_server = two_player_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # ~~~ ACT ~~~
    # create a client and connect to matching port
    context = zmq.Context()
    req_sock = ErrorCatchingSocket(context, zmq.REQ)
    req_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))

    # send engagement (which is out of context for drift phase) 
    # and get response
    req_msg = deepcopy(ENG_REQ_MSG_0)
    req_sock.send_json(req_msg)
    rep_msg = req_sock.recv_json()

    # ~~~ ASSERT ~~~
    # NOTE: this just checks that an error message is returned, not what's in the message
    # Althought the out-of-context is what is supposed to cause the error, that is not what 
    # is asserted here. It is possible the out-of-context error is not actually caught
    # but some other error IS caught. This would cause this test to give
    # misleading false positives
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'engagementPhase'
    assert 'error' in rep_msg.keys()
    assert 'message' in rep_msg['error'].keys()

def test_TwoPlayerGameServer_register_player(two_player_game_server_fixture):
    """ test that two, but not three, players can be registered
    """

    # ~~~ ARRANGE ~~~

    # start the python-unity server
    game = koth.KOTHGame(**DEFAULT_PARAMS)
    game_server = two_player_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # create a client and connect to matching port
    context = zmq.Context()
    plr_1_sock = ErrorCatchingSocket(context, zmq.REQ)
    plr_1_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))
    plr_2_sock = ErrorCatchingSocket(context, zmq.REQ)
    plr_2_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))

    # create a player registration request message
    req_msg = dict()
    req_msg['apiVersion'] = API_VER_NUM_2P
    req_msg['context'] = 'playerRegistration'
    req_msg['playerAlias'] = 'harry'
    # req_msg['data'] = dict()
    # req_msg['data']['kind'] = 'playerRegistrationRequest'
    # req_msg['data']['gamertag'] = 'harry'

    # ~~~ ACT ~~~

    # send registration requests in order
    # and get response
    plr_1_sock.send_json(req_msg)
    rep_msg = plr_1_sock.recv_json()
    game_id = rep_msg['gameID']

    # ~~~ ASSERT ~~~
    # check registration successful
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'playerRegistration'
    assert rep_msg['data']['kind'] == 'playerRegistrationResponse'
    assert rep_msg['data']['playerAlias'] == 'harry'
    assert rep_msg['data']['playerID'] == U.P1
    assert isinstance(rep_msg['data']['playerUUID'], str)
    assert 'error' not in rep_msg.keys()

    # try to re-register from the same client, causing an error
    plr_1_sock.send_json(req_msg)
    rep_msg = plr_1_sock.recv_json()
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'playerRegistration'
    assert rep_msg['gameID'] == game_id
    assert 'error' in rep_msg.keys()
    assert 'message' in rep_msg['error'].keys()
    assert 'data'  not in rep_msg.keys()

    # ~~~ REPEAT ~~~
    req_msg['playerAlias'] = 'draco'
    plr_2_sock.send_json(req_msg)
    rep_msg = plr_2_sock.recv_json()

    # check registration successful
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'playerRegistration'
    assert rep_msg['gameID'] == game_id
    assert rep_msg['data']['kind'] == 'playerRegistrationResponse'
    assert rep_msg['data']['playerAlias'] == 'draco'
    assert rep_msg['data']['playerID'] == U.P2
    assert isinstance(rep_msg['data']['playerUUID'], str)
    assert 'error' not in rep_msg.keys()

    # create a 3rd player socket and ensure returns error when attempting to register
    # since player slots are already full
    plr_3_sock = ErrorCatchingSocket(context, zmq.REQ)
    plr_3_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))
    req_msg['playerAlias'] = 'ron'
    plr_3_sock.send_json(req_msg)
    rep_msg = plr_3_sock.recv_json()
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'playerRegistration'
    assert rep_msg['gameID'] == game_id
    assert 'error' in rep_msg.keys()
    assert 'message' in rep_msg['error'].keys()
    assert 'data'  not in rep_msg.keys()


def test_TwoPlayerGameServer_handle_unfilled_queue(two_player_game_server_fixture):
    """ test that action request with unfilled queue returns a waiting response
    """

    # ~~~ ARRANGE ~~~
    # create game object with default params
    game = koth.KOTHGame(**DEFAULT_PARAMS)
    game.update_turn_phase(U.MOVEMENT)

    # start the python-unity server
    game_server = two_player_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # create a client and connect to matching port
    context = zmq.Context()
    plr1_sock = ErrorCatchingSocket(context, zmq.REQ)
    plr1_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))
    plr2_sock = ErrorCatchingSocket(context, zmq.REQ)
    plr2_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))

    # create a player registration request message
    registration_msg = dict()
    registration_msg['apiVersion'] = API_VER_NUM_2P
    registration_msg['context'] = 'playerRegistration'
    registration_msg['playerAlias'] = 'harry'
    plr1_sock.send_json(registration_msg)
    rep_msg = plr1_sock.recv_json()
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'playerRegistration'
    game_id = rep_msg['gameID']
    assert rep_msg['data']['kind'] == 'playerRegistrationResponse'
    assert rep_msg['data']['playerAlias'] == 'harry'
    assert rep_msg['data']['playerID'] == U.P1
    assert isinstance(rep_msg['data']['playerUUID'], str)
    assert 'error' not in rep_msg.keys()
    plr1_uuid = deepcopy(rep_msg['data']['playerUUID'])

    registration_msg['playerAlias'] = 'draco'
    plr2_sock.send_json(registration_msg)
    rep_msg = plr2_sock.recv_json()
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'playerRegistration'
    assert rep_msg['gameID'] == game_id
    assert rep_msg['data']['kind'] == 'playerRegistrationResponse'
    assert rep_msg['data']['playerAlias'] == 'draco'
    assert rep_msg['data']['playerID'] == U.P2
    assert isinstance(rep_msg['data']['playerUUID'], str)
    assert 'error' not in rep_msg.keys()
    plr2_uuid = deepcopy(rep_msg['data']['playerUUID'])

    # ~~~ ACT ~~~

    # send a dummy action request to get a wait message back
    act_msg = deepcopy(MOV_REQ_MSG_0)
    act_msg['apiVersion'] = API_VER_NUM_2P
    act_msg['playerAlias'] = 'harry'
    act_msg['playerUUID'] = plr1_uuid
    plr1_sock.send_json(act_msg)
    rep_msg = plr1_sock.recv_json()

    # ~~~ ASSERT ~~~
    # check that an unfilled queue message has been returned
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'movementPhase'
    assert 'gameID' in rep_msg.keys()
    assert rep_msg['data']['kind'] == 'waitingResponse'
    assert 'error' not in rep_msg.keys()

def test_TwoPlayerGameServer_handle_game_reset(two_player_game_server_fixture):
    """ test that mutually agreed game reset resets game state
    """

    # ~~~ ARRANGE ~~~
    # create game object with default params
    game = koth.KOTHGame(**DEFAULT_PARAMS)

    # start the python-unity server
    game_server = two_player_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # create clients and connect to matching port
    context = zmq.Context()
    plr1_sock = ErrorCatchingSocket(context, zmq.REQ)
    plr1_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))
    plr2_sock = ErrorCatchingSocket(context, zmq.REQ)
    plr2_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))

    # create a player registration request message
    registration_msg = dict()
    registration_msg['apiVersion'] = API_VER_NUM_2P
    registration_msg['context'] = 'playerRegistration'
    registration_msg['playerAlias'] = 'harry'
    plr1_sock.send_json(registration_msg)
    rep_msg = plr1_sock.recv_json()
    assert 'error' not in rep_msg.keys()
    plr1_uuid = deepcopy(rep_msg['data']['playerUUID'])

    registration_msg['playerAlias'] = 'draco'
    plr2_sock.send_json(registration_msg)
    rep_msg = plr2_sock.recv_json()
    assert 'error' not in rep_msg.keys()
    plr2_uuid = deepcopy(rep_msg['data']['playerUUID'])

    # ~~~ ACT ~~~

    # send game reset requests from first client
    act_msg = deepcopy(GAME_RESET_REQ_MSG)
    act_msg['apiVersion'] = API_VER_NUM_2P
    act_msg['playerAlias'] = 'harry'
    act_msg['playerUUID'] = plr1_uuid
    plr1_sock.send_json(act_msg)
    rep_msg = plr1_sock.recv_json()

    # ~~~ ASSERT ~~~
    
    # check that an unfilled queue message has been returned
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'gameReset'
    assert 'gameID' in rep_msg.keys()
    assert rep_msg['data']['kind'] == 'waitingResponse'
    assert 'error' not in rep_msg.keys()

    # send game reset request from second client
    act_msg['playerAlias'] = 'draco'
    act_msg['playerUUID'] = plr2_uuid
    plr2_sock.send_json(act_msg)
    rep_msg = plr2_sock.recv_json()

    # check that an advancing message is returned
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'gameReset'
    assert 'gameID' in rep_msg.keys()
    assert rep_msg['data']['kind'] == 'advancingResponse'
    assert 'error' not in rep_msg.keys()

def test_TwoPlayerGameServer_handle_inconsistent_player_uuid(two_player_game_server_fixture):
    """ test that client sending wrong uuid returns error
    """

    # ~~~ ARRANGE ~~~
    # create game object with default params
    game = koth.KOTHGame(**DEFAULT_PARAMS)
    game.update_turn_phase(U.MOVEMENT)

    # start the python-unity server
    game_server = two_player_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # create clients and connect to matching port
    context = zmq.Context()
    plr1_sock = ErrorCatchingSocket(context, zmq.REQ)
    plr1_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))
    plr2_sock = ErrorCatchingSocket(context, zmq.REQ)
    plr2_sock.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))

    # create a player registration request message
    registration_msg = dict()
    registration_msg['apiVersion'] = API_VER_NUM_2P
    registration_msg['context'] = 'playerRegistration'
    registration_msg['playerAlias'] = 'harry'
    plr1_sock.send_json(registration_msg)
    rep_msg = plr1_sock.recv_json()
    assert 'error' not in rep_msg.keys()
    plr1_uuid = deepcopy(rep_msg['data']['playerUUID'])

    registration_msg['playerAlias'] = 'draco'
    plr2_sock.send_json(registration_msg)
    rep_msg = plr2_sock.recv_json()
    assert 'error' not in rep_msg.keys()
    plr2_uuid = deepcopy(rep_msg['data']['playerUUID'])

    # ~~~ ACT ~~~

    # send a dummy action request with invalid uuid to get a wait message back
    act_msg = deepcopy(MOV_REQ_MSG_0)
    act_msg['apiVersion'] = API_VER_NUM_2P
    act_msg['playerAlias'] = 'harry'
    act_msg['playerUUID'] = str(uuid.uuid4())
    plr1_sock.send_json(act_msg)
    rep_msg = plr1_sock.recv_json()

    # ~~~ ASSERT ~~~
    assert rep_msg['apiVersion'] == API_VER_NUM_2P
    assert rep_msg['context'] == 'movementPhase'
    assert 'gameID' in rep_msg.keys()
    assert 'error' in rep_msg.keys()
    assert 'message' in rep_msg['error'].keys()
    assert 'data'  not in rep_msg.keys()

def notest_TwoPlayerGameServer_handle_move_phase(two_player_game_server_fixture):
    """ test that complete move phase request updates game state
    """

    # ~~~ ARRANGE ~~~
    # create game object with default params
    game = koth.KOTHGame(**DEFAULT_PARAMS)
    game.update_turn_phase(U.MOVEMENT)

    # start the python-unity server
    game_server = two_player_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # TODO: complete implementation
    assert False

def notest_TwoPlayerGameServer_handle_engage_phase(two_player_game_server_fixture):
    """ test that complete engage phase request updates game state
    """

    # ~~~ ARRANGE ~~~
    # create game object with default params
    game = koth.KOTHGame(**DEFAULT_PARAMS)
    game.update_turn_phase(U.MOVEMENT)

    # start the python-unity server
    game_server = two_player_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # TODO: complete implementation
    assert False

def notest_TwoPlayerGameServer_handle_drift_phase(two_player_game_server_fixture):
    """ test that complete drift phase request updates game state
    """

    # ~~~ ARRANGE ~~~
    # create game object with default params
    game = koth.KOTHGame(**DEFAULT_PARAMS)
    game.update_turn_phase(U.MOVEMENT)

    # start the python-unity server
    game_server = two_player_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # TODO: complete implementation
    assert False

def test_SingleUserGameServer_echo_request(single_user_game_server_fixture):
    """Tests if the echo server responds properly
    
    Ref:
        http://lists.idyll.org/pipermail/testing-in-python/2011-October/004507.html
    """

    # ~~~ ARRANGE ~~~
    # start the python-unity server
    game_server = single_user_game_server_fixture(None)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game_server

    # create a client and connect to matching port
    context = zmq.Context()
    req_sock = ErrorCatchingSocket(context, zmq.REQ)
    req_sock.connect("tcp://localhost:{}".format(PORT_NUM))

    # ~~~ ACT ~~~
    # create simple dictionary and send as json to be echoed
    req_msg = ECHO_REQ_MSG_0
    req_sock.send_json(req_msg)

    # get response and check for matching
    rep_msg = req_sock.recv_json()

    # ~~~ ASSERT ~~~
    assert rep_msg == ECHO_REQ_MSG_0

def test_SingleUserGameServer_handle_game_reset_request(single_user_game_server_fixture):
    """ test that a game reset requests response with init game state
    """

    # ~~~ ARRANGE ~~~
    # create game object with default params
    game = koth.KOTHGame(**DEFAULT_PARAMS)

    # start game server object
    game_server = single_user_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # ~~~ ACT ~~~
    # create a client and connect to matching port
    context = zmq.Context()
    req_sock = ErrorCatchingSocket(context, zmq.REQ)
    req_sock.connect("tcp://localhost:{}".format(PORT_NUM))

    # send request and get response
    req_sock.send_json(GAME_RESET_REQ_MSG)
    rep_msg = req_sock.recv_json()

    # ~~~ ASSERT ~~~
    assert rep_msg['apiVersion'] == API_VER_NUM
    assert rep_msg['context'] == 'gameReset'
    assert rep_msg['data']['kind'] == 'gameResetResponse'
    assert rep_msg['data']['gameState']['turnNumber'] == 0
    assert rep_msg['data']['gameState']['turnPhase'] == U.MOVEMENT


def test_SingleUserGameServer_handle_engagement_action_selection(single_user_game_server_fixture):
    """ test that a engagement selection responds with engagement outcomes
    """

    # create game object with 1 ring and 2 pieces per player and set phase to engagment
    game = koth.KOTHGame(
        max_ring=1, 
        min_ring=1, 
        geo_ring=1,
        init_board_pattern=INIT_BOARD_PATTERN_2,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_2, init_pattern_beta=INIT_BOARD_PATTERN_2)
    # game.game_state[U.TURN_PHASE] = U.ENGAGEMENT
    game.update_turn_phase(U.ENGAGEMENT)

    # start game server object
    game_server = single_user_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # create a client and connect to matching port
    context = zmq.Context()
    req_sock = ErrorCatchingSocket(context, zmq.REQ)
    req_sock.connect("tcp://localhost:{}".format(PORT_NUM))

    # send request and get response
    req_sock.send_json(ENG_REQ_MSG_0)
    rep_msg = req_sock.recv_json()

    # assertions on response message
    assert rep_msg['apiVersion'] == API_VER_NUM
    assert rep_msg['context'] == 'engagementPhase'
    assert rep_msg['data']['kind'] == 'engagementPhaseResponse'
    res_seq = rep_msg['data']['resolutionSequence']
    assert res_seq[0]['actionType'] == 'guard'
    assert res_seq[0]['attackerID'] == 'alpha:bludger:1'
    assert res_seq[0]['targetID'] == 'beta:seeker:0'
    assert res_seq[0]['guardianID'] == 'beta:bludger:1'
    assert res_seq[0]['probability'] == DGP.ENGAGE_PROBS[U.IN_SEC][U.GUARD]
    # Cannot test success since probabilistic

    assert res_seq[1]['actionType'] == 'shoot'
    assert res_seq[1]['attackerID'] == 'alpha:bludger:1'
    if res_seq[0]['success']:
        assert res_seq[1]['targetID'] == 'beta:bludger:1'
    else:
        assert res_seq[1]['targetID'] == 'beta:seeker:0'
    assert res_seq[1]['guardianID'] == ''
    assert res_seq[1]['probability'] == DGP.ENGAGE_PROBS[U.ADJ_SEC][U.SHOOT]
    # Cannot test success since probabilistic

    # assertions on game state
    # TODO

def test_SingleUserGameServer_handle_movement_action_selection(single_user_game_server_fixture):
    """ test that a movement selection responds with correct updated state
    """

    # create game object with 1 ring and 2 pieces per player and set phase to engagment
    game = koth.KOTHGame(
        max_ring=1, 
        min_ring=1, 
        geo_ring=1,
        init_board_pattern=INIT_BOARD_PATTERN_2,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_2, init_pattern_beta=INIT_BOARD_PATTERN_2)
    # game.game_state[U.TURN_PHASE] = U.MOVEMENT
    game.update_turn_phase(U.MOVEMENT)

    # start game server object
    game_server = single_user_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # create a client and connect to matching port
    context = zmq.Context()
    req_sock = ErrorCatchingSocket(context, zmq.REQ)
    req_sock.connect("tcp://localhost:{}".format(PORT_NUM))

    # send request and get response
    req_sock.send_json(MOV_REQ_MSG_0)
    rep_msg = req_sock.recv_json()

    # assertions on response message
    assert rep_msg['apiVersion'] == API_VER_NUM
    assert rep_msg['context'] == 'movementPhase'
    assert rep_msg['data']['kind'] == 'movementPhaseResponse'
    assert len(rep_msg['data']['gameState']['tokenStates']) == 4
    token_states = {v[GS.PIECE_ID]:{} for v in rep_msg['data']['gameState']['tokenStates']}
    # assert rep_msg['data']['alpha:seeker:0']['position'] == 1
    # assert rep_msg['data']['alpha:bludger:1']['position'] == 2
    # assert rep_msg['data']['beta:seeker:0']['position'] == 2
    # assert rep_msg['data']['beta:bludger:1']['position'] == 1

def test_SingleUserGameServer_handle_drift_request(single_user_game_server_fixture):
    """ test that a game advances past drift phase and responds with game state
    """

    # ~~~ ARRANGE ~~~
    # create game object with default params
    game = koth.KOTHGame(**DEFAULT_PARAMS)
    game.update_turn_phase(U.DRIFT)

    # start game server object
    game_server = single_user_game_server_fixture(game)
    game_server.start()

    # remove local game and game_server objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game, game_server

    # ~~~ ACT ~~~
    # create a client and connect to matching port
    context = zmq.Context()
    req_sock = ErrorCatchingSocket(context, zmq.REQ)
    req_sock.connect("tcp://localhost:{}".format(PORT_NUM))

    # send request and get response
    req_sock.send_json(DRIFT_REQ_MSG)
    rep_msg = req_sock.recv_json()

    # ~~~ ASSERT ~~~
    assert rep_msg['apiVersion'] == API_VER_NUM
    assert rep_msg['context'] == 'driftPhase'
    assert rep_msg['data']['kind'] == 'driftPhaseResponse'
    assert rep_msg['data']['gameState']['turnNumber'] == 1
    assert rep_msg['data']['gameState']['turnPhase'] == U.MOVEMENT


def test_game_id(single_user_game_server_fixture):
    ''' check memory address of game object for multiprocessing'''
    
    # create game, set game state
    game = koth.KOTHGame(
        max_ring=1, 
        min_ring=1, 
        geo_ring=1,
        init_board_pattern=INIT_BOARD_PATTERN_2,
        **DEFAULT_PARAMS_PARTIAL)
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_2, init_pattern_beta=INIT_BOARD_PATTERN_2)
    # game.game_state[U.TURN_PHASE] = U.ENGAGEMENT
    game.update_turn_phase(U.ENGAGEMENT)

    # create server, and start server
    game_server = single_user_game_server_fixture(game)
    game_server.start()

    # create a client and connect to matching port
    context = zmq.Context()
    req_sock = ErrorCatchingSocket(context, zmq.REQ)
    req_sock.connect("tcp://localhost:{}".format(PORT_NUM))

    # send request and get response
    req_sock.send_json(ENG_REQ_MSG_0)
    rep_msg = req_sock.recv_json()

    assert str(id(game)) == str(id(game_server.game))
    assert str(id(game)) != str(rep_msg['gameID'])


if __name__ == "__main__":
    test_SingleUserGameServer_echo_request(SingleUserGameServer(None, {'tcp_port': PORT_NUM}))
    # test_GameServer_echo_request(game_server_fixture)
