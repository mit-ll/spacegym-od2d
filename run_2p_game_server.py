#Need this to start the game server and register the alpah client. Another script will register the beta client.
#I could do this as a stand alone server program, but would have to make sure to send reset requests from teh clients when they join and not sure how to 
#Register new player does a reset on the game, so that could work...

import zmq
import threading
import numpy as np
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
import orbit_defender2d.king_of_the_hill.default_game_parameters as DGP
#import orbit_defender2d.king_of_the_hill.default_game_parameters_small as DGP
from orbit_defender2d.king_of_the_hill import koth
from orbit_defender2d.king_of_the_hill import game_server as GS
from orbit_defender2d.king_of_the_hill.examples.server_utils import *

from time import sleep


class ListenerClient(object):
    '''bundles REQ and SUB sockets in one object'''
    def __init__(self, router_addr, pub_addr, plr_alias, sub_topic=''):
        ''' Create req and sub socket, and a thread for subsciption handling
        Args:
            router_addr : str
                IP+port number for connection to server ROUTER
            pub_addr : str
                IP+port number for connection to server PUB
            plr_alias : str
                alias used for registered player in KOTH game
            sub_topic : str
                topic for SUB subscription

        Notes:
            Want to use threads, not multiple processes, because I wanted shared memory objects
        
        Refs:
            https://realpython.com/intro-to-python-threading/
            https://stackoverflow.com/questions/24843193/stopping-a-python-thread-running-an-infinite-loop
        '''

        super().__init__()

        ctx = zmq.Context()
        self.alias = plr_alias
        self.player_id = None
        self.game_state = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # establish REQ socket and connect to ROUTER
        self.req_socket = ctx.socket(zmq.REQ)
        self.req_socket.connect(router_addr)

        # establish SUB socket and connect to PUB
        self.sub_socket = ctx.socket(zmq.SUB)
        # must set a subscription, missing this step is a common mistake. 
        # https://zguide.zeromq.org/docs/chapter1/#Getting-the-Message-Out
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, sub_topic) 
        self.sub_socket.connect(pub_addr)

        # establish subscription thread
        # make daemon so it is killed when __main__ ends
        # sub_thread = threading.Thread(target=self.subscriber_func, daemon=True)
        sub_thread = threading.Thread(target=self.subscriber_func)
        sub_thread.start()

    def register_player_req(self):
        '''format player registration request, send req, recv response, and check'''

        # format registration request message
        req_msg = dict()
        req_msg['apiVersion'] = API_VER_NUM_2P
        req_msg['context'] = 'playerRegistration'
        req_msg['playerAlias'] = self.alias

        # send registration request
        self.req_socket.send_json(req_msg)
        rep_msg = self.req_socket.recv_json()

        # check registration successful
        assert rep_msg['apiVersion'] == API_VER_NUM_2P
        assert rep_msg['context'] == 'playerRegistration'
        assert rep_msg['data']['kind'] == 'playerRegistrationResponse'
        assert rep_msg['data']['playerAlias'] == self.alias
        assert rep_msg[GS.DATA][GS.PLAYER_ID] in [U.P1, U.P2]
        assert isinstance(rep_msg[GS.DATA][GS.PLAYER_UUID], str)
        assert 'error' not in rep_msg.keys()

        # record backend player id
        self.player_id = rep_msg[GS.DATA][GS.PLAYER_ID]
        self.player_uuid = rep_msg[GS.DATA][GS.PLAYER_UUID]
    
    def assert_consistent_registry(self, registry):
        '''check that registry has not changed unexpectedly'''
        reg_entry = [reg for reg in registry if reg[GS.PLAYER_ALIAS]==self.alias]
        assert len(reg_entry) == 1
        reg_entry = reg_entry[0]
        assert reg_entry[GS.PLAYER_ID] == self.player_id, "Expect ID {}, got {}".format(self.player_id, reg_entry[GS.PLAYER_ID])

    def game_reset_req(self):
        '''format game reset request, send request, recv response, and check'''

        # format game reset request message
        req_msg = dict()
        req_msg['apiVersion'] = API_VER_NUM_2P
        req_msg['context'] = 'gameReset'
        req_msg['playerAlias'] = self.alias
        req_msg['playerUUID'] = self.player_uuid

        # send game reset request
        self.req_socket.send_json(req_msg)
        rep_msg = self.req_socket.recv_json()

        # check reset waiting or advancing
        assert rep_msg['apiVersion'] == API_VER_NUM_2P
        assert rep_msg['context'] == 'gameReset'
        assert rep_msg['data']['kind'] in ['waitingResponse', 'advancingResponse']
        assert 'error' not in rep_msg.keys()

    def send_random_action_req(self, context):
        ''' format and send random-yet-legal action depending on context '''
        req_msg = dict()
        req_msg['apiVersion'] = API_VER_NUM_2P
        req_msg['playerAlias'] = self.alias
        req_msg['playerUUID'] = self.player_uuid

        if context == U.DRIFT:
            req_msg['context'] = 'driftPhase'

        else:
            # select random valid action formatted as client request dictionary
            plr_actions = []
            req_msg[GS.DATA] = dict()
            for tok in self.game_state[GS.TOKEN_STATES]:
                if koth.parse_token_id(tok[GS.PIECE_ID])[0] == self.player_id:
                    #act = tok[GS.LEGAL_ACTIONS][choice(len(tok[GS.LEGAL_ACTIONS]))]
                    act = tok[GS.LEGAL_ACTIONS][0]
                    act[GS.PIECE_ID] = tok[GS.PIECE_ID]
                    plr_actions.append(act)

            if context == U.MOVEMENT:
                req_msg[GS.CONTEXT] = GS.MOVE_PHASE
                req_msg[GS.DATA][GS.KIND] = GS.MOVE_PHASE_REQ
                req_msg[GS.DATA][GS.MOVEMENT_SELECTIONS] = plr_actions

            elif context == U.ENGAGEMENT:
                req_msg[GS.CONTEXT] = GS.ENGAGE_PHASE
                req_msg[GS.DATA][GS.KIND] = GS.ENGAGE_PHASE_REQ
                req_msg[GS.DATA][GS.ENGAGEMENT_SELECTIONS] = plr_actions

            else:
                raise ValueError

        # send game reset request
        self.req_socket.send_json(req_msg)
        rep_msg = self.req_socket.recv_json()

        # check reset waiting or advancing
        assert rep_msg[GS.API_VERSION] == API_VER_NUM_2P
        assert rep_msg[GS.CONTEXT] in [GS.DRIFT_PHASE, GS.MOVE_PHASE, GS.ENGAGE_PHASE]
        assert 'error' not in rep_msg.keys(), "error received: {}".format(rep_msg[GS.ERROR][GS.MESSAGE])
        assert rep_msg[GS.DATA][GS.KIND] in [GS.WAITING_RESP, GS.ADVANCING_RESP]
            

    def drift_phase_req(self):
        '''format drift request, send msg, recv response, and check'''

        # format drift request
        req_msg = dict()
        req_msg['apiVersion'] = API_VER_NUM_2P
        req_msg['context'] = 'driftPhase'
        req_msg['playerAlias'] = self.alias
        req_msg['playerUUID'] = self.player_uuid

        # send drift request
        self.req_socket.send_json(req_msg)
        rep_msg = self.req_socket.recv_json()

        # check reset waiting or advancing
        assert rep_msg['apiVersion'] == API_VER_NUM_2P
        assert rep_msg['context'] == 'driftPhase'
        assert rep_msg['data']['kind'] in ['waitingResponse', 'advancingResponse']
        assert 'error' not in rep_msg.keys()


    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def subscriber_func(self):
        '''wait for and process message published on PUB
        
        Refs:
            https://stackoverflow.com/questions/26012132/zero-mq-socket-recv-call-is-blocking
        '''

        while not self.stopped():

            try:

                # wait for published message
                msg = self.sub_socket.recv_json(flags=zmq.NOBLOCK)

                # check message content
                assert msg[GS.API_VERSION] == API_VER_NUM_2P, "expected {}, got {}".format(API_VER_NUM_2P, msg[GS.API_VERSION])
                assert GS.ERROR not in msg.keys()

                # if registry response, wait a little while for request socket in other thread to 
                # to have time to receive registry info and update client info
                if msg[GS.CONTEXT] == GS.PLAYER_REGISTRATION:
                    sleep(0.25)

                # verify registry and update game state (shared memory, therefore use a lock)
                with self._lock:
                    #self.assert_consistent_registry(msg[GS.DATA][GS.PLAYER_REGISTRY])
                    self.game_state = msg[GS.DATA][GS.GAME_STATE]
                    assert_valid_game_state(game_state=self.game_state)

                print('{} client received and processed message on SUB!'.format(self.alias))

            except zmq.Again as e:
                # no messages waiting to be processed
                pass


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
    max_turns = DGP.MAX_TURNS,
    fuel_points_factor_bludger = DGP.FUEL_POINTS_FACTOR_BLUDGER)


ROUTER_PORT_NUM = 5555
PUB_PORT_NUM = 5556

API_VER_NUM_2P = "v2022.07.26.0000.2p"

ECHO_REQ_MSG_0 = {'context': 'echo', 'data': {'key0': 'value0'}}

def start_server():
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

    return game_server

def create_listener_client():
    #Register a listening client to monitor the game
    listener_client = ListenerClient(
        router_addr="tcp://localhost:{}".format(ROUTER_PORT_NUM),
        pub_addr="tcp://localhost:{}".format(PUB_PORT_NUM),
        plr_alias='listener_client',)
    return listener_client

def run_listener(game_server, listener_client):   
    # Don't register the client as a player, just subscribe to the pub socket
    tmp_game_state = listener_client.game_state
    game_started = False
    game_finised = False

    while tmp_game_state is None:
        sleep(3)
        tmp_game_state = listener_client.game_state
        print("Waiting for a game to begin")
        
    game_started = True
    print("Game started")

    while tmp_game_state['gameDone'] is False:
        sleep(3)
        tmp_game_state = listener_client.game_state
        print("Waiting for game to finish")
    
    game_finised = True
    print("Game finished")
    
    restart_server(game_server, listener_client)

def restart_server(game_server, listener_client):
    #Restart the game server and listener client
    #Stop the listener client if it exists
    listener_client.stop()
    del listener_client

    game_server.terminate()
    game_server.join()
    del game_server

    game_server = start_server()
    listener_client = create_listener_client()
    run_listener(game_server, listener_client)

if __name__ == "__main__":
    game_server = start_server()
    listener_client = create_listener_client()
    run_listener(game_server, listener_client)
