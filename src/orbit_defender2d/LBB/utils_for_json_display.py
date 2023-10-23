# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.game_server as GS
from orbit_defender2d.king_of_the_hill.koth import KOTHGame

import json
from collections import namedtuple
from typing import Dict, List, Tuple

CUR_API_VERSION = "v2022.02.02.0000.display"



def format_action_message(
        data_kind: str, 
        context: str,
        game: KOTHGame,
        actions: Dict) -> Dict:
        ''' Format game state into API-Compatible dictionary to be sent as response message

        Args:
            req_msg (dict): request message converted to dictionary
            data_kind (str): descriptor of data kind
            game_state (dict): API-compatible list of game state
            engagement_outcomes (list): API-compatible list of sequence of engagement outcomes
        
        Returns:
            rep_msg (dict): API-compatible response message containing game state
        '''

        act_msg = dict()
        act_msg[GS.API_VERSION] = CUR_API_VERSION
        act_msg[GS.CONTEXT] = context
        act_msg['gameID'] = id(game)
        act_msg[GS.DATA] = dict()
        act_msg[GS.DATA][GS.KIND] = data_kind
        if data_kind == GS.ENGAGE_PHASE_REQ:
            
            engagement_selections = [{
                GS.PIECE_ID:i[0], 
                GS.ACTION_TYPE:i[1][0],
                GS.TARGET_ID: i[1][1]} for i in actions.items()]

            act_msg[GS.DATA][GS.ENGAGEMENT_SELECTIONS] = engagement_selections
        elif data_kind == GS.MOVE_PHASE_REQ:
            
            movement_selections = [{
                GS.PIECE_ID:i[0], 
                GS.ACTION_TYPE:i[1][0]} for i in actions.items()]

            act_msg[GS.DATA][GS.MOVEMENT_SELECTIONS] = movement_selections

        return act_msg

def format_response_message(
        data_kind: str, 
        context: str,
        game: KOTHGame) -> Dict:
        ''' Format game state into API-Compatible dictionary to be sent as response message

        Args:
            req_msg (dict): request message converted to dictionary
            data_kind (str): descriptor of data kind
            game_state (dict): API-compatible list of game state
            engagement_outcomes (list): API-compatible list of sequence of engagement outcomes
        
        Returns:
            rep_msg (dict): API-compatible response message containing game state
        '''

        rep_msg = dict()
        rep_msg[GS.API_VERSION] = CUR_API_VERSION
        rep_msg[GS.CONTEXT] = context
        rep_msg['gameID'] = id(game)
        rep_msg[GS.DATA] = dict()
        rep_msg[GS.DATA][GS.KIND] = data_kind
        rep_msg[GS.DATA][GS.GAME_STATE] = get_game_state(game.game_state, game.token_catalog)
        if data_kind == GS.ENGAGE_PHASE_RESP:
            
            engagement_outcomes = [{
                GS.ACTION_TYPE:i.action_type, 
                GS.ATTACKER_ID:i.attacker, 
                GS.TARGET_ID:i.target,
                GS.GUARDIAN_ID:i.guardian, 
                GS.PROB:i.prob,
                GS.SUCCESS:i.success} for i in game.engagement_outcomes]

            rep_msg[GS.DATA][GS.RESOLUTION_SEQUENCE] = engagement_outcomes

        return rep_msg

def get_game_state(koth_game_state, koth_token_catalog):
        ''' encode game state and engagement outcomes as API-compatible dictionaries
        '''

        game_state = dict()
        game_state[GS.TURN_NUMBER] = koth_game_state[U.TURN_COUNT]
        game_state[GS.TURN_PHASE] = koth_game_state[U.TURN_PHASE]
        game_state[GS.GAME_DONE] = koth_game_state[U.GAME_DONE]
        game_state[GS.GOAL_ALPHA] = koth_game_state[U.GOAL1]
        game_state[GS.GOAL_BETA] = koth_game_state[U.GOAL2]
        game_state[GS.SCORE_ALPHA] = koth_game_state[U.P1][U.SCORE]
        game_state[GS.SCORE_BETA] = koth_game_state[U.P2][U.SCORE]
        game_state[GS.TOKEN_STATES] = [{
            GS.PIECE_ID:token_name,
            GS.FUEL:token_state.satellite.fuel,
            GS.ROLE:token_state.role,
            GS.POSITION:token_state.position,
            GS.AMMO:token_state.satellite.ammo,
            GS.LEGAL_ACTIONS:get_token_legal_actions(koth_game_state, token_name=token_name)
            } for token_name, token_state in koth_token_catalog.items()]

        return game_state

def get_token_legal_actions(koth_game_state, token_name):
        ''' get list of dictionaries of legal actions from game state'''
        legal_actions = None
        if koth_game_state[U.TURN_PHASE] == U.MOVEMENT:
            legal_actions = [{
                GS.ACTION_TYPE:i.action_type} for i in koth_game_state[U.LEGAL_ACTIONS][token_name]]
        elif koth_game_state[U.TURN_PHASE] == U.ENGAGEMENT:
            legal_actions = [{
                GS.ACTION_TYPE:i.action_type, 
                GS.TARGET_ID:i.target} for i in koth_game_state[U.LEGAL_ACTIONS][token_name]]
        elif koth_game_state[U.TURN_PHASE] == U.DRIFT:
            legal_actions = []
        else:
            raise ValueError("Unrecognized turn phase {}".format(koth_game_state[U.TURN_PHASE]))

        return legal_actions
