# Try to make a 2-player game with 1 random player and 1 remote (human) player.
# This uses the two plaeyr game server "GS.TwoPlayerGameServer(game=game, comm_configs=comm_configs)"
# Is the right way to do this is probably to utilize the single player game server, so that the AI agent
#has access to the whole game space and can mvoe when the human player is ready?
#   - no, it seems like i need to use the two player gamer server and just figure out how to sync the game states
#   

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
#API_VER_NUM_2P = "v2021.11.18.0000.2p"
API_VER_NUM_2P = "v2022.07.26.0000.2p"
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

#Define the AI player client 
class PlayerClient(object):
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
                    self.assert_consistent_registry(msg[GS.DATA][GS.PLAYER_REGISTRY])
                    self.game_state = msg[GS.DATA][GS.GAME_STATE]
                    assert_valid_game_state(game_state=self.game_state)

                print('{} client received and processed message on SUB!'.format(self.alias))

            except zmq.Again as e:
                # no messages waiting to be processed
                pass


def run_server_2p_1ran_1remote_game():
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

    #Create and connect a random player client
    print("Creating alpha client...")
    alpha_client = PlayerClient(
        router_addr="tcp://localhost:{}".format(ROUTER_PORT_NUM),
        pub_addr="tcp://localhost:{}".format(PUB_PORT_NUM),
        plr_alias='harry'
    )
    #Register the alpha player
    print("Registering alpha client with alias {}".format(alpha_client.alias))
    alpha_client.register_player_req()

    #Send game reset request and wait for human player to join
    alpha_client.game_reset_req()

    cur_game_state = alpha_client.game_state

    #Wait for the human player to connect
    while cur_game_state is None:
        print("Waiting on human player to connect to router port {}".format(ROUTER_PORT_NUM))
        cur_game_state = alpha_client.game_state
        sleep(5)

    while not cur_game_state[GS.GAME_DONE]:
        
        print("\n<==== Turn: {} | Phase: {} ====>".format(cur_game_state[GS.TURN_NUMBER], cur_game_state[GS.TURN_PHASE]))
        # send random, legal, context-dependent action from the random client
        turnphase = cur_game_state[GS.TURN_PHASE]
        print('{} ({}) client sending action request'.format(
            alpha_client.alias, alpha_client.player_id))
        alpha_client.send_random_action_req(context=cur_game_state[GS.TURN_PHASE])
        
        # wait for game state to advance
        while cur_game_state[GS.TURN_PHASE] == turnphase and not cur_game_state[GS.GAME_DONE]:
            sleep(5)
            cur_game_state = alpha_client.game_state
            print('waiting for turn phase {} to advance'.format(cur_game_state[GS.TURN_PHASE]))

    # cleanup
    print("Terminating server...")
    game_server.terminate()
    game_server.join()

    winner_id = None
    winner_alias = None
    alpha_score = cur_game_state[GS.SCORE_ALPHA]
    beta_score = cur_game_state[GS.SCORE_BETA]
    if alpha_score > beta_score:
        winner_id = U.P1
        winner_alias = alpha_client.alias
    elif beta_score > alpha_score:
        winner_id = U.P2
        winner_alias = "Beta"
    else:
        winner_id = 'draw'

    if cur_game_state[GS.TOKEN_STATES][0]['fuel'] <= DGP.MIN_FUEL:
        term_cond = "alpha out of fuel"
    elif cur_game_state[GS.TOKEN_STATES][1]['fuel'] <= DGP.MIN_FUEL:
        term_cond = "beta out of fuel"
    elif cur_game_state[GS.SCORE_ALPHA] >= DGP.WIN_SCORE:
        term_cond = "alpha reached Win Score"
    elif cur_game_state[GS.SCORE_BETA]  >= DGP.WIN_SCORE:
        term_cond = "beta reached Win Score"
    elif cur_game_state[GS.TURN_NUMBER]  >= DGP.MAX_TURNS:
        term_cond = "max turns reached" 
    else:
        term_cond = "unknown"

    print(
        "\n====GAME FINISHED====\n" +
        "Winner: {} ({})\n".format(winner_alias, winner_id) + 
        "Score: {}|{}\n".format(alpha_score, beta_score) + 
        "=====================\n")
    print("Termination condition: {}".format(term_cond))
    return



if __name__ == "__main__":
    run_server_2p_1ran_1remote_game()
