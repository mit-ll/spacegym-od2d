# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# Communication bridge to Unity for rendering and human I/O using ZMQ

import zmq
import json
import multiprocessing
import uuid

from bidict import bidict
from tornado import ioloop
from zmq.eventloop import zmqstream
from typing import Dict, List, Tuple
from collections import namedtuple
from copy import deepcopy

import orbit_defender2d.utils.utils as U
from orbit_defender2d.king_of_the_hill.koth import KOTHGame

TCP_PORT = 'tcp_port'
ROUTER_PORT = 'router_port'
PUB_PORT = 'publisher_port'

# ref: https://google.github.io/styleguide/jsoncstyleguide.xml
# ref: https://github.com/mit-ll/spacegym-od2d/wiki/Dev-Notes#python-unity-json-api-v202105030000
RESERVED_TOP_LEVEL_NAMES = [API_VERSION, CONTEXT, DATA, ERROR] = ['apiVersion', 'context', 'data', 'error']
RESERVED_DATA_NAMES = [KIND, ITEMS] = ['kind', 'items']
RESERVED_ERROR_NAMES = [CODE, MESSAGE] = ['code', 'message']
MSG_CONTEXTS = \
    [GAME_RESET, MOVE_PHASE, ENGAGE_PHASE, DRIFT_PHASE, ECHO, PLAYER_REGISTRATION] = \
    ['gameReset', 'movementPhase', 'engagementPhase', 'driftPhase', 'echo', 'playerRegistration'] 
MSG_KINDS = \
    [GAME_RESET_RESP, DRIFT_PHASE_RESP, 
    MOVE_PHASE_REQ, MOVE_PHASE_RESP, 
    ENGAGE_PHASE_REQ, ENGAGE_PHASE_RESP,
    PLAYER_REGISTRATION_REQ, PLAYER_REGISTRATION_RESP,
    WAITING_RESP, ADVANCING_RESP] = \
    ['gameResetResponse', 'driftPhaseResponse', 
    'movementPhaseRequest', 'movementPhaseResponse', 
    'engagementPhaseRequest', 'engagementPhaseResponse',
    'playerRegistrationRequest', 'playerRegistrationResponse',
    'waitingResponse', 'advancingResponse']
PLAYER_REGISTRY = 'playerRegistry'
GAME_STATE = 'gameState'
GAME_STATE_FIELDS = [TURN_NUMBER, TURN_PHASE, GAME_DONE, GOAL_ALPHA, GOAL_BETA, SCORE_ALPHA, SCORE_BETA, TOKEN_STATES] = \
    ['turnNumber', 'turnPhase', 'gameDone', 'goalSectorAlpha', 'goalSectorBeta', 'scoreAlpha', 'scoreBeta', 'tokenStates']
TOKEN_STATE_FIELDS = [PIECE_ID, FUEL, ROLE, POSITION, AMMO, LEGAL_ACTIONS] = \
    ['pieceID', 'fuel', 'role', 'position', 'ammo', 'legalActions']

MOVEMENT_SELECTIONS = 'movementSelections'
ENGAGEMENT_SELECTIONS = 'engagementSelections'
RESOLUTION_SEQUENCE = 'resolutionSequence'
ACTION_TYPE = 'actionType'
TARGET_ID = 'targetID'
ATTACKER_ID = 'attackerID'
GUARDIAN_ID = 'guardianID'
SUCCESS = 'success'
PROB = 'probability'
# SERVER_ID = 'serverID'
GAME_ID = 'gameID'
PLAYER_ID = 'playerID'
PLAYER_ALIAS = 'playerAlias'
PLAYER_UUID = 'playerUUID'

# these version numbers define the functionality and message
# formatting that will be returned by GameServer objects in this module.
# Any requests sent to the GameServer objects must match these api version
# numbers, otherwise the request might be misinterpreted or cause an error
CUR_1P_API_VERSION = "v2021.11.18.0000.1p"
CUR_2P_API_VERSION = "v2022.07.26.0000.2p"

# RegisteredPlayer = namedtuple('RegisteredPlayer', ['player_id', 'client_uid'])
ClientIDTuple = namedtuple('ClientIDTuple', ['alias', 'uid'])

# class PlayerRegistry:
#     ''' maintains one-to-one mapping between player IDs and (alias, client_uid) tuples'''
#     def __init__(self):
#         self.register = {}
#     def add(self, pid, alias_uid):
#         ''' add 1-to-1 mapping to register
#         Args:
#             id : str
#                 backend player identifier (e.g. alpha, beta)
#             alias_uid : ClientIDTuple
#                 tuple of user-select player name (alias) and ROUTER-assigned client UID
#         '''
#         self.register[alias_uid] = pid
#         self.register[pid] = alias_uid
#     def remove(self, k):
#         ''' remove entry from register
#         Args:
#             k : str OR ClientIDTuple
#                 backend player identifier OR tuple of (alias, uid)
#         '''
#         self.register.pop(self.register.pop(k))
#     def get(self, k):
#         return self.d[k]

class GameServer(multiprocessing.Process):
    ''' Brokers interactions between human (via Unity) and AI (via Gym) via ZMQ socket

    This is to be used for:
        1. Human-v-AI games
        2. Human-v-Human games 
    This is NOT to be used for:
        1. AI-v-AI training 
        2. AI-v-AI visualization

    Args:
        game (KothAlphaGame): game object that defines game state and rules
            Note: currently this is specified as a KOTHALphaGame, but this could abstracted in 
            in the future
        comm_configs (Dict): configuration params to establish TCP connection to unity frontend
    '''
    def __init__(self, game: KOTHGame, comm_configs: Dict) -> None:
        super().__init__()
        self.comm_configs = comm_configs
        self.game = game

    def run(self):
        raise NotImplementedError('Child class must implement run()')

    def echo_request(self, req_msg) -> Dict:
        ''' a simple function to test TCP connection with Unity client 
        
        Args:
            req_msg (dict): dictionary converted from json message

        Returns:
            rep_msg (dict): echo's back the req_msg

        Ref:
            https://zeromq.org/get-started/?language=python&library=pyzmq#
        '''
        return req_msg

    def format_game_state_response_message(self, 
        req_msg: Dict,
        api_version: str, 
        data_kind: str, 
        game_state: Dict, 
        engagement_outcomes: List,
        is_2player: bool=False) -> Dict:
        ''' Format game state into API-Compatible dictionary to be sent as response message

        Args:
            req_msg (dict): request message converted to dictionary
            api_version (str): api version number 
            data_kind (str): descriptor of data kind
            game_state (dict): API-compatible list of game state
            engagement_outcomes (list): API-compatible list of sequence of engagement outcomes
            is_2player (bool): if True, include player registry
        
        Returns:
            rep_msg (dict): API-compatible response message containing game state
        '''

        rep_msg = dict()
        rep_msg[API_VERSION] = api_version
        rep_msg[CONTEXT] = req_msg[CONTEXT]
        rep_msg[GAME_ID] = id(self.game)
        rep_msg[DATA] = dict()
        rep_msg[DATA][KIND] = data_kind
        rep_msg[DATA][GAME_STATE] = game_state
        if data_kind == ENGAGE_PHASE_RESP:
            rep_msg[DATA][RESOLUTION_SEQUENCE] = engagement_outcomes

        if is_2player:
            assert api_version.split('.')[-1] == '2p', "Expected 2-player API, got {}".format(api_version)
            reg = []
            for plr_id, cli_id in self.player_registry.items():
                reg.append({PLAYER_ID: plr_id, PLAYER_ALIAS: cli_id.alias})
            rep_msg[DATA][PLAYER_REGISTRY] = reg

        return rep_msg

    def get_game_state(self):
        ''' encode game state and engagement outcomes as API-compatible dictionaries
        '''

        game_state = dict()
        game_state[TURN_NUMBER] = self.game.game_state[U.TURN_COUNT]
        game_state[TURN_PHASE] = self.game.game_state[U.TURN_PHASE]
        game_state[GAME_DONE] = self.game.game_state[U.GAME_DONE]
        game_state[GOAL_ALPHA] = self.game.game_state[U.GOAL1]
        game_state[GOAL_BETA] = self.game.game_state[U.GOAL2]
        game_state[SCORE_ALPHA] = self.game.game_state[U.P1][U.SCORE]
        game_state[SCORE_BETA] = self.game.game_state[U.P2][U.SCORE]
        game_state[TOKEN_STATES] = [{
            PIECE_ID:token_name,
            FUEL:token_state.satellite.fuel,
            ROLE:token_state.role,
            POSITION:token_state.position,
            AMMO:token_state.satellite.ammo,
            LEGAL_ACTIONS:self.get_token_legal_actions(token_name=token_name)
            } for token_name, token_state in self.game.token_catalog.items()]

        return game_state

    def get_token_legal_actions(self, token_name):
        ''' get list of dictionaries of legal actions from game state'''
        legal_actions = None
        if self.game.game_state[U.TURN_PHASE] == U.MOVEMENT:
            legal_actions = [{
                ACTION_TYPE:i.action_type} for i in self.game.game_state[U.LEGAL_ACTIONS][token_name]]
        elif self.game.game_state[U.TURN_PHASE] == U.ENGAGEMENT:
            legal_actions = [{
                ACTION_TYPE:i.action_type, 
                TARGET_ID:i.target} for i in self.game.game_state[U.LEGAL_ACTIONS][token_name]]
        elif self.game.game_state[U.TURN_PHASE] == U.DRIFT:
            legal_actions = []
        else:
            raise ValueError("Unrecognized turn phase {}".format(self.game.game_state[U.TURN_PHASE]))

        return legal_actions

class TwoPlayerGameServer(GameServer):
    ''' game server that assumes two users on separate clients
    '''

    def __init__(self, game: KOTHGame, comm_configs: Dict) -> None:
        super().__init__(game=game, comm_configs=comm_configs)
        self.router_stream = None   # stream for handling action requests from player clients
        self.player_registry = bidict()
        self.reset_player_input_queue() 

    def run(self):
        ''' Setup and run ROUTER and PUB sockets to handle I/O from player clients
        Refs: 
            http://lists.idyll.org/pipermail/testing-in-python/2011-October/004507.html
            https://zguide.zeromq.org/docs/chapter3/#The-Extended-Reply-Envelope
        '''

        # setup zmq context used for all sockets
        ctx = zmq.Context()

        # prepare publisher socket to send state information
        self.publisher_socket = ctx.socket(zmq.PUB)
        self.publisher_socket.bind("tcp://*:{}".format(self.comm_configs[PUB_PORT]))

        # create I/O loop to accept requests to router
        router_loop = ioloop.IOLoop.instance()

        # create ROUTER socket and stream for handling actions requests from player clients
        router_socket = ctx.socket(zmq.ROUTER)
        router_socket.bind("tcp://*:{}".format(self.comm_configs[ROUTER_PORT]))
        self.router_stream = zmqstream.ZMQStream(router_socket, router_loop)
        self.router_stream.on_recv(self.router_io)

        # start router loop
        router_loop.start()

    def reset_player_input_queue(self):
        ''' resets player inputs to None
        '''
        self.player_input_queue = {U.P1: None, U.P2: None}

    def router_io(self, raw_msg:List) -> None:
        ''' Top-level input/output interface of request message to router
        unpack message, send request to game-based callback, route response to client

        Args:
            raw_msg : list(bytes)
                list of byte strings representing json

        Returns:
            None
                
        Ref:
            https://pyzmq.readthedocs.io/en/latest/api/zmq.eventloop.zmqstream.html#zmq.eventloop.zmqstream.ZMQStream.on_recv
        '''

        # extract connection id and request message from respective frames
        # Ref: https://zguide.zeromq.org/docs/chapter3/#The-Extended-Reply-Envelope
        connection_id = raw_msg[0]
        req_msg = json.loads(raw_msg[2])

        # get response message
        resp_msg = self.process_request(req_msg=req_msg)

        # send response message 
        # Need to use multipart message with appropriate frames to respond to a REQ socket
        # json must be converted to string then bytes
        # Ref: https://zguide.zeromq.org/docs/chapter3/#The-Extended-Reply-Envelope
        # Ref: https://zguide.zeromq.org/docs/chapter3/#ROUTER-Broker-and-REQ-Workers
        self.router_stream.send_multipart([
            connection_id,
            b'',
            json.dumps(resp_msg).encode('utf-8')
        ])
    
    def process_request(self, req_msg:Dict) -> None:
        ''' send request to appropriate callback and return response message

        Args:
            req_msg: dict
                player registration request message
        
        Returns:
            resp_msg: dict
                player registration response or error
        '''

        # handle message response based on message context
        if req_msg[CONTEXT] == ECHO:

            # simply echo back message, regardless of content
            return self.echo_request(req_msg)


        # check api version
        err_msg = self.check_api_version(req_msg=req_msg)
        if err_msg:
            return err_msg

        # if no api version error, handle message based on context
        if req_msg[CONTEXT] == PLAYER_REGISTRATION:

            # generate unique client id
            client_uid = str(uuid.uuid4())

            # register new player if slot is available, return error if not
            resp_msg, start_game = self.register_new_player(client_uid, req_msg)

            # start game if both players registered (condition decided by register func)
            if start_game:
                
                # reset game, access and format game state data
                self.game.reset_game()
                game_state = self.get_game_state()
                engagement_outcomes = None

                # publish new game state on PUB socket
                pub_msg = self.format_game_state_response_message(
                    req_msg = req_msg,
                    api_version=CUR_2P_API_VERSION,
                    data_kind=GAME_RESET_RESP,
                    game_state=game_state, 
                    engagement_outcomes=engagement_outcomes,
                    is_2player=True)
                self.publisher_socket.send_json(pub_msg)

            return resp_msg

        elif req_msg[CONTEXT] in [GAME_RESET, MOVE_PHASE, ENGAGE_PHASE, DRIFT_PHASE]:

            # verify request player alias matches existing registered player
            cli_id = ClientIDTuple(alias=req_msg[PLAYER_ALIAS], uid=req_msg[PLAYER_UUID])
            if cli_id not in self.player_registry.inv.keys():

                # no player registered with this alias-UID combination, return error
                return self.handle_invalid_request(req_msg=req_msg,
                    err_str="No player registered with alias {} and REQ client ID {}".format(
                        cli_id.alias,
                        cli_id.uid
                    )
                )
            
            # map client id information to backend player identifier (e.g. U.P1 OR U.P2)
            player_id = self.player_registry.inv[cli_id]

            # apply player's request to take action in game object 
            return self.process_game_action_request(req_msg=req_msg, player_id=player_id)

        else:
            return self.handle_invalid_request(req_msg=req_msg,
                    err_str="Unrecognized message context {}".format(req_msg[CONTEXT]))

        raise NotImplementedError('Expected to reach a return statement before this')


    def register_new_player(self, client_uid: str, req_msg:Dict) -> Dict:
        ''' handle new player registration request, return error if no slot available

        Args:
            client_uid : str
                unique identity assigned to client by server
            req_msg : dict
                player registration request message
        
        Returns:
            resp_msg : dict
                player registration response or error
            start_game : bool 
                if true, game is to be reset and game state published

        '''

        # check there is no data entry, per the API
        if DATA in req_msg.keys():
            return self.handle_invalid_request(
                req_msg=req_msg,
                err_str='No {} object expected in request context {}'.format(
                    DATA, req_msg[CONTEXT]
                )
            ), False

        # create client ID tuple tat stores the user-selected alias and ROUTER-assigned UID
        cli_id = ClientIDTuple(
            alias=req_msg[PLAYER_ALIAS], 
            uid=client_uid)

        # check for player alias collisions
        for cid in self.player_registry.inv.keys():
            if cid.alias == cli_id.alias:
                return self.handle_invalid_request(
                    req_msg=req_msg,
                    err_str='Client with alias {} already registered to player {}'.format(
                        cli_id.alias, self.player_registry.inv[cid]
                    )
                ), False

        # begin formatting response
        resp_msg = dict()
        resp_msg[API_VERSION] = CUR_2P_API_VERSION
        resp_msg[CONTEXT] = req_msg[CONTEXT]
        resp_msg[GAME_ID] = id(self.game)   # python identifier of game object
        resp_msg[DATA] = dict()
        resp_msg[DATA][KIND] = PLAYER_REGISTRATION_RESP
        
        # register new players to empty slots in order of arrival
        start_game = False
        if U.P1 not in self.player_registry.keys():
            plr_id = U.P1

        elif U.P2 not in self.player_registry.keys():
            plr_id = U.P2
            start_game = True

        else:
            # send error message if both player slots are already full
            return self.handle_invalid_request(
                req_msg=req_msg,
                err_str='No player slots available in game'
            ), False

        # register player in game server
        # plr_alias = req_msg[PLAYER_ALIAS]
        self.player_registry[plr_id] = cli_id

        # format response with backend player id to send to client
        resp_msg[DATA][PLAYER_ALIAS] = self.player_registry[plr_id].alias
        resp_msg[DATA][PLAYER_ID] = plr_id
        resp_msg[DATA][PLAYER_UUID] = self.player_registry[plr_id].uid

        return resp_msg, start_game
            
    def process_game_action_request(self, req_msg:Dict, player_id) -> Dict:
        ''' Integrate client action requests and update game state

        Args:
            req_msg: dict
                dictionary formatting the request message
            player_id: str
                backend player identifier (e.g. alpha, beta)

        Returns:
            resp_msg: dict
                dictionary formatting the response message
        '''

        # check player id is valid and label the other player
        assert player_id in [U.P1, U.P2]
        other_player_id = U.P2 if player_id==U.P1 else U.P1

        # verify context of message matches game phase, send error msg if not
        err_msg = self.check_game_context(req_msg=req_msg)
        if err_msg:
            return err_msg

        # store message as player-specific, game-phase-appropriate request
        # TODO: send error message if player-speicifc request already queued
        # err_msg = self.update_player_inputs(req_msg)
        self.player_input_queue[player_id] = req_msg

        # if both player requests are present
        if self.player_input_queue_filled():

            # check for matching context for both player inputs
            # NOTE: check_game_context only checks one request, not matching request from both players
            cntx = self.player_input_queue[player_id][CONTEXT]
            if cntx != self.player_input_queue[other_player_id][CONTEXT]:
                return self.handle_invalid_request(req_msg=req_msg,
                    err_str="Mis-matched player request contexts\n" + \
                        "player {}: {}\n".format(player_id, cntx) + \
                        "player {}: {}\n".format(other_player_id, self.player_input_queue[other_player_id][CONTEXT]))

            # handle game reset case
            if cntx == GAME_RESET:
                
                # reset game state
                self.game.reset_game()
                data_kind = GAME_RESET_RESP

            elif cntx in [MOVE_PHASE, ENGAGE_PHASE, DRIFT_PHASE]:

                # integrate both player requests into a complete verbose action
                player_actions, data_kind = self.synthesize_verbose_action()

                # apply the verbose action to the game to update game state
                self.game.apply_verbose_actions(actions=player_actions)

            else:

                # inproper message contexts should have already been handled gracefully at
                # this point, raising error un-gracefully in case something slipped through the cracks
                raise ValueError("Unexpected player request contexts: {}".format(cntx))


            # access and format game state data
            game_state = self.get_game_state()
            engagement_outcomes = None
            if cntx == ENGAGE_PHASE:
                engagement_outcomes = [{
                    ACTION_TYPE:i.action_type, 
                    ATTACKER_ID:i.attacker, 
                    TARGET_ID:i.target,
                    GUARDIAN_ID:i.guardian, 
                    PROB:i.prob,
                    SUCCESS:i.success} for i in self.game.engagement_outcomes]

            # publish new game state on PUB socket
            resp_msg = self.format_game_state_response_message(
                req_msg = req_msg,
                api_version=CUR_2P_API_VERSION,
                data_kind=data_kind,
                game_state=game_state, 
                engagement_outcomes=engagement_outcomes,
                is_2player=True)
            self.publisher_socket.send_json(resp_msg)

            # reset player inputs
            self.reset_player_input_queue()

            # return trivial response message indicating game state has advanced
            # Note: does not send game state in this message, that must be collected
            # by the new message sent on the PUB socket
            return self.format_response_with_data(
                req_msg=req_msg, 
                resp_data={KIND: ADVANCING_RESP})
        
        else:
            # form response message informs player client that we are still
            # waiting on other player's request
            return self.format_response_with_data(
                req_msg=req_msg, 
                resp_data={KIND: WAITING_RESP})

        # return response message
        raise NotImplementedError('Expected to reach a return statement before this')

    def synthesize_verbose_action(self):
        '''Integrate requests form both players to get a single, verbose action for game

        Returns:
            actions (dict): verbose action description
                            key is piece id token_catalog, one for each piece in game
                            value is the piece's movement tuple (U.MOVEMENT_TYPES) or 
                            engagement tuple (ENGAGMENT_TYPE, target_piece_id, prob)

            data_kind (str) : string description of data to be sent in response
        '''

        # assert that both actions exist and match
        # NOTE: non-matching request should already be handled more gracefully
        # with an error message to the client in calling functions like 
        # process_game_action_request, but catching here with an assert just in
        # case something slips through
        req_p1 = self.player_input_queue[U.P1]
        req_p2 = self.player_input_queue[U.P2]
        cntx = req_p1[CONTEXT]
        assert req_p2[CONTEXT] == cntx

        actions = None
        data_kind = None
        if cntx == DRIFT_PHASE:
            # no actions are taken in drift phase
            # game object expects None
            data_kind = DRIFT_PHASE_RESP

        elif cntx == MOVE_PHASE:

            # extract movements from json message and format as dictionary of MovementTuples
            # NOTE: does not check for conflicting or missing token ID's, this should have
            # been handled gracefully by check_player_request_tokens but the data *could* have
            # been altered once stored in player_input_queue
            move_reqs = req_p1[DATA][MOVEMENT_SELECTIONS]+req_p2[DATA][MOVEMENT_SELECTIONS]
            actions = {v[PIECE_ID]:U.MovementTuple(action_type=v[ACTION_TYPE]) for v in move_reqs}
            data_kind = MOVE_PHASE_RESP

        elif cntx == ENGAGE_PHASE:
            
            # extract movements from json message and format as dictionary of MovementTuples
            # NOTE: does not check for conflicting or missing token ID's, this should have
            # been handled gracefully by check_player_request_tokens but the data *could* have
            # been altered once stored in player_input_queue
            engage_reqs = req_p1[DATA][ENGAGEMENT_SELECTIONS]+req_p2[DATA][ENGAGEMENT_SELECTIONS]
            actions = {v[PIECE_ID]:U.EngagementTuple(
                action_type=v[ACTION_TYPE], 
                target=v[TARGET_ID], 
                prob=self.game.get_engagement_probability(
                    token_id=v[PIECE_ID], 
                    target_id=v[TARGET_ID], 
                    engagement_type=v[ACTION_TYPE])) \
                for v in engage_reqs}
            data_kind = ENGAGE_PHASE_RESP

        else:
            raise ValueError("Unexpected context {}".format(cntx))

        return actions, data_kind

    def format_response_with_data(self, req_msg: Dict, resp_data: Dict) -> Dict:
        ''' package a response message given a dict for the data subfield

        Args:
            req_msg: dict
                dictionary formatting the request message
            resp_data: dict
                dictionary of data field in response message

        Returns:
            resp_msg: dict
                dictionary formatting the response message

        '''
        resp_msg = dict()
        resp_msg[API_VERSION] = CUR_2P_API_VERSION
        resp_msg[CONTEXT] = req_msg[CONTEXT]
        resp_msg[GAME_ID] = id(self.game)
        resp_msg[DATA] = dict()
        resp_msg[DATA] = resp_data

        return resp_msg

    def player_input_queue_filled(self) -> bool:
        '''check for inputs from both players'''
        return all([
            (p in self.player_input_queue.keys() and self.player_input_queue[p] is not None)
            for p in [U.P1, U.P2]
        ])


    def handle_invalid_request(self, req_msg:Dict, err_str:str) -> Dict:
        ''' Format response to invalid request

        Args:
            req_msg: dict
                dictionary formatting the request message
            err_str: str
                string description of invalid request

        Returns:
            resp_msg: dict
                dictionary formatting the response message
        '''

        resp_msg = dict()
        resp_msg[API_VERSION] = CUR_2P_API_VERSION
        resp_msg[CONTEXT] = req_msg[CONTEXT]
        resp_msg[GAME_ID] = id(self.game)
        resp_msg[ERROR] = dict()
        resp_msg[ERROR][MESSAGE] = "Invalid Request: {}".format(err_str)

        return resp_msg

    def check_game_context(self, req_msg:Dict):
        ''' verify message context and kind align with game state
        Args:
            req_msg: dict
                dictionary formatting the request message

        Returns:
            err_msg: None OR dict
                dictionary formatting the response message
        '''
        is_err = False
        err_msg = None
        err_str = ""

        def err_data_kind_formatter(expected_kind):
            return "In context {}, expected data of kind {}. Got {}\n".format(
                req_msg[CONTEXT],
                expected_kind,
                req_msg[DATA][KIND])

        def err_game_state_formatter(expected_game_state):
            return "In context {}, expected a game state of {}. Got {}\n".format(
                req_msg[CONTEXT],
                expected_game_state,
                self.game.game_state[U.TURN_PHASE])

        if req_msg[CONTEXT] in [GAME_RESET]:
            pass
        elif req_msg[CONTEXT] == MOVE_PHASE:
            tok_check = self.check_player_request_tokens(req_msg=req_msg)
            if tok_check:
                is_err = True
                err_str += tok_check
            if self.game.game_state[U.TURN_PHASE] != U.MOVEMENT:
                is_err = True
                err_str += err_game_state_formatter(U.MOVEMENT)
            if req_msg[DATA][KIND] != MOVE_PHASE_REQ:
                is_err = True
                err_str += err_data_kind_formatter(MOVE_PHASE_REQ)
        elif req_msg[CONTEXT] == ENGAGE_PHASE:
            tok_check = self.check_player_request_tokens(req_msg=req_msg)
            if tok_check:
                is_err = True
                err_str += tok_check
            if self.game.game_state[U.TURN_PHASE] != U.ENGAGEMENT:
                is_err = True
                err_str += err_game_state_formatter(U.ENGAGEMENT)
            if req_msg[DATA][KIND] != ENGAGE_PHASE_REQ:
                is_err = True
                err_str += err_data_kind_formatter(ENGAGE_PHASE_REQ)
        elif req_msg[CONTEXT] == DRIFT_PHASE:
            if self.game.game_state[U.TURN_PHASE] != U.DRIFT:
                is_err = True
                err_str += err_game_state_formatter(U.DRIFT)
        else:
            is_err = True
            err_str += "Unexpected message context {}".format(req_msg[CONTEXT])

        if is_err:
            err_msg = self.handle_invalid_request(req_msg=req_msg, err_str=err_str)

        return err_msg

    def check_api_version(self, req_msg: Dict):
        ''' verify api version
        Args:
            req_msg: dict
                dictionary formatting the request message

        Returns:
            err_msg: None OR dict
                dictionary formatting the response message
        '''
        err_msg = None

        if req_msg[API_VERSION] != CUR_2P_API_VERSION:
            err_msg = self.handle_invalid_request(
                req_msg=req_msg,
                err_str="Invalid API version!\nExpected: {}\nReceived: {}".format(
                    CUR_2P_API_VERSION, req_msg[API_VERSION]
                )
            )
        
        return err_msg

    def check_player_request_tokens(self, req_msg: Dict):
        ''' verify that player requests actions for all, and only, their tokens
        Args:
            req_msg: dict
                dictionary formatting the request message

        Returns:
            err_str: None OR dict
                string describing error, if present
        '''
        # TODO: complete implementation
        return None

class SingleUserGameServer(GameServer):
    ''' game server that assumes a single client for the user interface
    '''

    def __init__(self, game: KOTHGame, comm_configs: Dict) -> None:
        super().__init__(game=game, comm_configs=comm_configs)
        self.server_stream = None

    def run(self):
        '''
        Ref: http://lists.idyll.org/pipermail/testing-in-python/2011-October/004507.html
        '''
        context = zmq.Context()
        loop = ioloop.IOLoop.instance()

        server_socket = context.socket(zmq.REP)
        server_socket.bind("tcp://*:{}".format(self.comm_configs[TCP_PORT]))
        self.server_stream = zmqstream.ZMQStream(server_socket, loop)
        self.server_stream.on_recv(self.handle_request)

        loop.start()

    def handle_request(self, raw_msg):
        ''' unpack message and route request to proper callback based on kind 

        Args:
            msg (bytes): list of byte string representing json from zmq recv_multipart() 
                
        Ref:
            https://pyzmq.readthedocs.io/en/latest/api/zmq.eventloop.zmqstream.html#zmq.eventloop.zmqstream.ZMQStream.on_recv
        '''

        # decode json message into dictionary
        req_msg = json.loads(raw_msg[0])

        # handle message response based on message kind
        if req_msg[CONTEXT] == ECHO:
            rep_msg = self.echo_request(req_msg)
        elif req_msg[CONTEXT] == GAME_RESET:
            assert req_msg[API_VERSION] == CUR_1P_API_VERSION
            rep_msg = self.handle_game_reset_request(req_msg)
        elif req_msg[CONTEXT] == MOVE_PHASE:
            assert req_msg[API_VERSION] == CUR_1P_API_VERSION
            assert req_msg[DATA][KIND] == MOVE_PHASE_REQ
            assert self.game.game_state[U.TURN_PHASE] == U.MOVEMENT
            rep_msg = self.handle_movement_action_selection(req_msg)
        elif req_msg[CONTEXT] == ENGAGE_PHASE:
            assert req_msg[API_VERSION] == CUR_1P_API_VERSION
            assert req_msg[DATA][KIND] == ENGAGE_PHASE_REQ
            assert self.game.game_state[U.TURN_PHASE] == U.ENGAGEMENT
            rep_msg = self.handle_engagement_action_selection(req_msg)
        elif req_msg[CONTEXT] == DRIFT_PHASE:
            assert req_msg[API_VERSION] == CUR_1P_API_VERSION
            assert self.game.game_state[U.TURN_PHASE] == U.DRIFT
            rep_msg = self.handle_drift_request(req_msg)
        else:
            raise ValueError("Unrecognized message context {}".format(req_msg[CONTEXT]))

        # send response message
        self.server_stream.send_json(rep_msg)


    def handle_game_reset_request(self, init_req_msg: Dict) -> Dict:
        ''' call game reset function and respond with new game state
        
        Args:
            init_req_msg (dict): dict converted from json message

        Returns:
            rep_msg (dict): new game state in api-compatible dictionary
        '''
        self.game.reset_game()

        rep_msg = self.format_game_state_response_message(
            req_msg = init_req_msg,
            api_version=CUR_1P_API_VERSION,
            data_kind=GAME_RESET_RESP,
            game_state=self.get_game_state(), 
            engagement_outcomes=None)

        return rep_msg

    def handle_drift_request(self, init_req_msg: Dict) -> Dict:
        ''' advance game through drift phase and respond with game state
        
        Args:
            init_req_msg (dict): dict converted from json message

        Returns:
            rep_msg (dict): new game state in api-compatible dictionary
        '''

        game_state = self.apply_selected_actions(actions=None)

        rep_msg = self.format_game_state_response_message(
            req_msg = init_req_msg,
            api_version=CUR_1P_API_VERSION,
            data_kind=DRIFT_PHASE_RESP,
            game_state=game_state, 
            engagement_outcomes=None)

        return rep_msg

    def handle_movement_action_selection(self, mov_req_msg: Dict) -> Dict:
        ''' update game state based on selected engagements and respond with new game state and engagement outcomes

        Args:
            move_req_msg (dict): dictionary converted from json message containing movement selections

        Returns:
            rep_msg (dict): new game state in api-compatible dictionary
        '''

        # extract movements from json message and format as dictionary of MovementTuples
        # movements = {k:mov_req_msg[DATA][MOVEMENT_SELECTIONS][k] for k in self.game.token_catalog.keys()}
        movements = {v[PIECE_ID]:U.MovementTuple(action_type=v[ACTION_TYPE]) \
            for v in mov_req_msg[DATA][MOVEMENT_SELECTIONS]}

        # apply engagement actions to update game state, get game state as
        # API-compatible dict
        game_state = self.apply_selected_actions(actions=movements)

        rep_msg = self.format_game_state_response_message(
            req_msg = mov_req_msg,
            api_version=CUR_1P_API_VERSION,
            data_kind=MOVE_PHASE_RESP,
            game_state=game_state, 
            engagement_outcomes=None)

        return rep_msg

    def handle_engagement_action_selection(self, eng_req_msg: Dict) -> Dict:
        ''' update game state based on selected engagements and respond with new game state and engagement outcomes

        Args:
            eng_msg (dict): dictionary converted from json message containing engagement sections

        Returns:
            rep_msg (dict): new game state and engagement outcomes in api-compatible dictionary
        '''

        # extract enagements from json message and format as dictionary of EngagementTuples
        engagements = {v[PIECE_ID]:U.EngagementTuple(
            action_type=v[ACTION_TYPE], 
            target=v[TARGET_ID], 
            prob=self.game.get_engagement_probability(
                token_id=v[PIECE_ID], 
                target_id=v[TARGET_ID], 
                engagement_type=v[ACTION_TYPE])) \
            for v in eng_req_msg[DATA][ENGAGEMENT_SELECTIONS]}

        # apply engagement actions to update game state, get game state as
        # API-compatible dict
        game_state = self.apply_selected_actions(actions=engagements)

        # encode engagement outcomes as API-compatible list
        engagement_outcomes = [{
            ACTION_TYPE:i.action_type, 
            ATTACKER_ID:i.attacker, 
            TARGET_ID:i.target,
            GUARDIAN_ID:i.guardian, 
            PROB:i.prob,
            SUCCESS:i.success} for i in self.game.engagement_outcomes]

        rep_msg = self.format_game_state_response_message(
            req_msg = eng_req_msg,
            api_version=CUR_1P_API_VERSION,
            data_kind=ENGAGE_PHASE_RESP,
            game_state=game_state, 
            engagement_outcomes=engagement_outcomes)

        return rep_msg

    def apply_selected_actions(self, actions: Dict) -> Dict:
        ''' Use selected actions to update and return the game state

        Args:
            actions (dict): verbose action description
                            key is piece id token_catalog, one for each piece in game
                            value is the piece's movement tuple (U.MOVEMENT_TYPES) or 
                            engagement tuple (ENGAGMENT_TYPE, target_piece_id, prob)
                    
        Returns:
            game_state (dict): state of game pieces formatted to the JSON API for response
        '''

        # apply engagement actions to update game state
        self.game.apply_verbose_actions(actions=actions)

        return self.get_game_state()


##### Game Server Utility functions #####

def assert_valid_game_state(game_state):
    '''check response from game server gives valid game state
    '''
    
    # turn counter
    assert isinstance(game_state[TURN_NUMBER], int)
    assert game_state[TURN_NUMBER] >= 0

    # turn phase
    assert game_state[TURN_PHASE] in U.TURN_PHASE_LIST

    # game done
    assert isinstance(game_state[GAME_DONE], bool)

    # goal positions
    assert isinstance(game_state[GOAL_ALPHA], int)
    # assert 0 < game_state[GS.GOAL_ALPHA] <= game_board.n_sectors
    assert isinstance(game_state[GOAL_BETA], int)
    # assert 0 < game_state[GS.GOAL_BETA] <= game_board.n_sectors

def print_game_info(game_state):
    '''
    Print the game state information from the game server.
    This is the game server version of game state, so not compatible with the kothgame game state.
    '''
    print("STATES:")
    for tok in game_state[TOKEN_STATES]:
        print("   {:<16s}| position: {:<4d}| fuel: {:<8.1f} ".format(tok[PIECE_ID], tok[POSITION], tok[FUEL]))
    print("alpha|beta score: {}|{}".format(game_state[SCORE_ALPHA],game_state[SCORE_BETA]))

def print_engagement_outcomes_list(engagement_outcomes):
    '''
    The engagement outcomes from the game server are a list of dicts instead of a list of named tuples like the kothgame engagement outcomes.
    See print_engagement_outcomes in koth.py for the kothgame version.
    '''
    print("ENGAGEMENT OUTCOMES:")
    # if engagement_outcomes is empty print No engagements
    if not engagement_outcomes:
        print("    No engagements")
    else:
        # print the engagement outcomes for guarding actions first
        print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format("Action", "Attacker", "Guardian", "Target", "Result"))
        for egout in engagement_outcomes:
            success_status = "Success" if egout[SUCCESS] else "Failure"
            if egout[ACTION_TYPE] == U.SHOOT or egout[ACTION_TYPE] == U.COLLIDE:
                print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format(
                    egout[ACTION_TYPE], egout[ATTACKER_ID], "", egout[TARGET_ID], success_status))
            elif egout[ACTION_TYPE] == U.GUARD:
                if isinstance(egout[ATTACKER_ID], str):
                    print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format(
                        egout[ACTION_TYPE], egout[ATTACKER_ID], egout[GUARDIAN_ID], egout[TARGET_ID], success_status))
                else:
                    print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format(
                        egout[ACTION_TYPE], "", egout[GUARDIAN_ID], egout[TARGET_ID], success_status))
            elif egout[ACTION_TYPE] == U.NOOP:
                print("NOOP")
            else:
                raise ValueError("Unrecognized action type {}".format(egout[ACTION_TYPE]))

def print_endgame_status(cur_game_state):
    '''
    Print the endgame scores, winner, and termination condition.
    '''

    winner = None
    #alpha_score = penv.kothgame.game_state[U.P1][U.SCORE]
    #beta_score = penv.kothgame.game_state[U.P2][U.SCORE]
    alpha_score = cur_game_state[SCORE_ALPHA]
    beta_score = cur_game_state[SCORE_BETA]
    
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'
    print("\n====GAME FINISHED====\nWinner: {}\nScore: {}|{}\n=====================\n".format(winner, alpha_score, beta_score))

    # The following requires the input arguments to the koth game, so it is better to get this from a similar koth print function.
    # if cur_game_state[TOKEN_STATES][0]['fuel'] <= DGP.MIN_FUEL:
    #     term_cond = "alpha out of fuel"
    # elif cur_game_state[TOKEN_STATES][1]['fuel'] <= DGP.MIN_FUEL:
    #     term_cond = "beta out of fuel"
    # elif cur_game_state[SCORE_ALPHA] >= DGP.WIN_SCORE[U.P1]:
    #     term_cond = "alpha reached Win Score"
    # elif cur_game_state[SCORE_BETA]  >= DGP.WIN_SCORE[U.P2]:
    #     term_cond = "beta reached Win Score"
    # elif cur_game_state[TURN_NUMBER]  >= DGP.MAX_TURNS:
    #     term_cond = "max turns reached" 
    # else:
    #     term_cond = "unknown"
    # print("Termination condition: {}".format(term_cond))

