# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import os
import datetime
import numpy as np
import networkx as nx
import orbit_defender2d.utils.utils as U
from copy import deepcopy
from collections import namedtuple, OrderedDict
from typing import Dict, List, Tuple
from orbit_defender2d.utils.orbit_grid import OrbitGrid
from orbit_defender2d.utils.satellite import Satellite
from orbit_defender2d.utils.engagement_graph import EngagementGraph

# Encode game parameters as named tuple so that
# 1. all parameters are grouped easily together
# 2. parameters are immutable within a KOTHGame instance
KOTHGameInputArgs = namedtuple('KOTHGameInputArgs',[
    'max_ring', 
    'min_ring', 
    'geo_ring',
    'init_board_pattern_p1',
    'init_board_pattern_p2',
    'init_fuel',
    'init_ammo',
    'min_fuel',
    'fuel_usage',
    'engage_probs',
    'illegal_action_score',
    'in_goal_points',
    'adj_goal_points',
    'fuel_points_factor',
    'win_score',
    'max_turns',
    'fuel_points_factor_bludger',]
)

class KOTHTokenState:
    ''' object encodes state of a single piece on the board (e.g. satellite, sector, etc.)'''
    def __init__(self, satellite, role, position):
        self.satellite = satellite
        self.role = role
        self.position = position

class KOTHGame:
    ''' object that encodes the state and rules of the King-of-the-Hill (KOTH) game
    
    Note this is distinguished from the environment env and raw_env which act as the 
    rigidly structured OpenAI gym interfaces for AI agent interactions within the game

    Compare with:
        CheckersRules: 
            https://github.com/PettingZoo-Team/PettingZoo/blob/1.6.1/pettingzoo/classic/checkers/checkers.py#L65
            https://github.com/PettingZoo-Team/PettingZoo/blob/1.6.1/pettingzoo/classic/checkers/checkers.py#L258
        chess.Board: 
            https://github.com/PettingZoo-Team/PettingZoo/blob/1.6.1/pettingzoo/classic/chess/chess_env.py#L27
            https://python-chess.readthedocs.io/en/latest/core.html#board
        go.Position:  
            https://github.com/PettingZoo-Team/PettingZoo/blob/1.6.1/pettingzoo/classic/go/go_env.py#L116
            https://github.com/PettingZoo-Team/PettingZoo/blob/1.6.1/pettingzoo/classic/go/go.py#L289
    '''
    def __init__(self, 
        max_ring: int, 
        min_ring: int, 
        geo_ring: int,
        init_board_pattern_p1: List,
        init_board_pattern_p2: List,
        init_fuel: Dict, 
        init_ammo: Dict, 
        min_fuel: float, 
        fuel_usage: Dict, 
        engage_probs: Dict, 
        illegal_action_score: float, 
        in_goal_points: Dict, 
        adj_goal_points: Dict, 
        fuel_points_factor: Dict, 
        win_score: Dict, 
        max_turns: int,
        fuel_points_factor_bludger: Dict ):
        '''
        Args:
            max_ring : int
                the highest orbital ring in board
            min_ring : int
                the lowest orbtial ring in board
            geo_ring : int
                the ring treated as stationary relative to earth
            init_board_pattern : List
                List of token placements relative to each players goal (hill).
                Each element is a tuple with first element is relative azim to hill
                and second element is number of bludger tokens to place there
            init_fuel : Dict
                Dict of initial fuel amounts (float values) for each token type 
            init_ammo : Dict
                Dict of initial ammo amounts (int values) for each token type
            min_fuel : float
                minimum fuel amount before satellite is inoperable
            fuel_usage : Dict
                fuel usage for each maneuver
            engage_probs : Dict
                probability of success for each engagement type
            illegal_action_score : float
                final score given to agent who chooses an illegal action
            in_goal_points: Dict
                points scored per timestep by agent who has seeker inside their goal (hill) sector
            adj_goal_points: Dict
                points scored per timestep by agent who has seeker adjacent to their goal (hill) sector
            fuel_points_factor: Dict
                conversion factor for computing points earned for seeker remaining fuel at each turn
            win_score: float
                if either player reaches this score, game is terminated and player wins (draw if both reach at same time)
            max_turns: int
                game is terminated if this number of turns is reached and winner is evaluated from current score
            fuel_points_factor_bludger: Dict
                conversion factor for computing points earned for bludger remaining fuel at each turn
        '''

        # check valid board definition
        assert 0 < min_ring <= geo_ring <= max_ring

        # encode input args as immutable namedtuple
        self.inargs = KOTHGameInputArgs(
            max_ring=max_ring,
            min_ring=min_ring,
            geo_ring=geo_ring,
            init_board_pattern_p1=init_board_pattern_p1,
            init_board_pattern_p2=init_board_pattern_p2,
            init_fuel=init_fuel,
            init_ammo=init_ammo,
            min_fuel=min_fuel,
            fuel_usage=fuel_usage,
            engage_probs=engage_probs,
            illegal_action_score=illegal_action_score,
            in_goal_points=in_goal_points,
            adj_goal_points=adj_goal_points,
            fuel_points_factor=fuel_points_factor,
            win_score=win_score,
            max_turns=max_turns,
            fuel_points_factor_bludger=fuel_points_factor_bludger,
        )

        # member variables derived from input args or elsewhere
        self.board_grid = OrbitGrid(n_rings=self.inargs.max_ring)
        self.player_names = [U.P1, U.P2]
        self.n_players = len(self.player_names)
        self.engagement_outcomes = None
        self.reset_game()

    def reset_game(self):
        ''' reset game state without reinstantiating a new game object
        '''
        self.game_state, self.token_catalog, self.n_tokens_alpha, self.n_tokens_beta = \
            self.initial_game_state(
                init_pattern_alpha=self.inargs.init_board_pattern_p1, 
                init_pattern_beta=self.inargs.init_board_pattern_p2)
        #update initial fuel score and score
        self.game_state[U.P1][U.SCORE] = self.get_fuel_points(player_id=U.P1) #Score track based on goal sector and fuel points
        self.game_state[U.P1][U.FUEL_SCORE] = self.get_fuel_points(player_id=U.P1) #Score track based on fuel remaining
        self.game_state[U.P2][U.SCORE] = self.get_fuel_points(player_id=U.P2) #Score track based on goal sector and fuel points
        self.game_state[U.P2][U.FUEL_SCORE] = self.get_fuel_points(player_id=U.P2) #Score track based on fuel remaining

    def terminate_game(self):
        ''' set game to done and return difference in score as reward
        '''
        self.game_state[U.GAME_DONE] = True

        # No need to update score here. It is updated in drift phase, which is the only place that terminate_game is called
        # update final score from fuel remaining
        #for plr_id in [U.P1, U.P2]:
        #    self.game_state[plr_id][U.SCORE] += self.get_fuel_points(player_id=plr_id)

        score_diff = self.game_state[U.P1][U.SCORE] - self.game_state[U.P2][U.SCORE]
        return {U.P1: score_diff, U.P2: -score_diff}

    def initial_game_state(self, 
        init_pattern_alpha: List, 
        init_pattern_beta: List) -> Tuple:
        ''' returns initial board configuration of pieces 

        Args:
            init_pattern_alpha (List): list of tuples for player alpha, 
                [0] is position relative to "hill", [1] is number of tokens to place there
            init_pattern_beta (List): list of tuples for player beta, 
                [0] is position relative to "hill", [1] is number of tokens to place there
            
        Returns:
            game_state (Dict): state of game and tokens within game
            token_catalog (Dict): token states with token names as keys
            n_token_alpha (int): number of tokens for player alpha
            n_token_beta (int): number of tokens for player beta 
        '''
        game_state = {U.P1:dict(), U.P2:dict()}
        token_catalog = OrderedDict()
        
        # Specify goal locations (i.e. "hills") in geo 180 degrees offset
        n_sectors_in_geo = self.board_grid.get_num_sectors_in_ring(self.inargs.geo_ring)
        goal1_azim = 0
        goal2_azim = n_sectors_in_geo//2
        p1_hill = self.board_grid.sector_coord2num(self.inargs.geo_ring, goal1_azim)
        game_state[U.GOAL1] = p1_hill
        p2_hill = self.board_grid.sector_coord2num(self.inargs.geo_ring, goal2_azim)
        game_state[U.GOAL2] = p2_hill

        # Populate the seeker pieces at team target sectors (hills)
        p1_state = [None]
        p1_state[0] = token_catalog[U.P1 + U.TOKEN_DELIMITER + U.SEEKER + U.TOKEN_DELIMITER + '0'] = \
            KOTHTokenState(
                Satellite(fuel=self.inargs.init_fuel[U.P1][U.SEEKER], ammo=self.inargs.init_ammo[U.P1][U.SEEKER]), 
                role=U.SEEKER, 
                position=p1_hill)
        n_tokens_alpha = 1

        p2_state = [None]
        p2_state[0] = token_catalog[U.P2 + U.TOKEN_DELIMITER + U.SEEKER + U.TOKEN_DELIMITER + '0'] = \
            KOTHTokenState(
                Satellite(fuel=self.inargs.init_fuel[U.P2][U.SEEKER], ammo=self.inargs.init_ammo[U.P2][U.SEEKER]), 
                role=U.SEEKER, 
                position=p2_hill)
        n_tokens_beta = 1

        # Populate team bludger pieces based on init_pattern relative to target sectors (hills)
        for init_val in init_pattern_alpha:
            rel_azim, n_sats = init_val
            for sat_i in range(n_sats):
                p1_state.append(None)
                p1_state[-1] = token_catalog[U.P1 + U.TOKEN_DELIMITER + U.BLUDGER + U.TOKEN_DELIMITER + str(n_tokens_alpha)] = \
                    KOTHTokenState(
                        Satellite(fuel=self.inargs.init_fuel[U.P1][U.BLUDGER], ammo=self.inargs.init_ammo[U.P1][U.BLUDGER]), 
                        role=U.BLUDGER, 
                        position=self.board_grid.get_relative_azimuth_sector(p1_hill, rel_azim))
                n_tokens_alpha += 1

        for init_val in init_pattern_beta:
            rel_azim, n_sats = init_val
            for sat_i in range(n_sats):
                p2_state.append(None)
                p2_state[-1] = token_catalog[U.P2 + U.TOKEN_DELIMITER + U.BLUDGER + U.TOKEN_DELIMITER + str(n_tokens_beta)] = \
                    KOTHTokenState(
                        Satellite(fuel=self.inargs.init_fuel[U.P2][U.BLUDGER], ammo=self.inargs.init_ammo[U.P2][U.BLUDGER]), 
                        role=U.BLUDGER, 
                        position=self.board_grid.get_relative_azimuth_sector(p2_hill, rel_azim))
                n_tokens_beta += 1

        #Figure out which player has more satellites, if any and add the difference as "removed" satellites with 0 fuel and ammo in position 0
        removed_sat_count = np.abs(n_tokens_alpha - n_tokens_beta)
        if removed_sat_count > 0:
            if n_tokens_alpha < n_tokens_beta:
                for sat_i in range(removed_sat_count):
                    p1_state.append(None)
                    p1_state[-1] = token_catalog[U.P1 + U.TOKEN_DELIMITER + U.BLUDGER + U.TOKEN_DELIMITER + str(n_tokens_alpha)] = \
                        KOTHTokenState(
                            Satellite(fuel=0, ammo=0), 
                            role=U.BLUDGER, 
                            position=0)
                    n_tokens_alpha += 1
            else:
                for sat_i in range(removed_sat_count):
                    p2_state.append(None)
                    p2_state[-1] = token_catalog[U.P2 + U.TOKEN_DELIMITER + U.BLUDGER + U.TOKEN_DELIMITER + str(n_tokens_beta)] = \
                        KOTHTokenState(
                            Satellite(fuel=0, ammo=0), 
                            role=U.BLUDGER, 
                            position=0)
                    n_tokens_beta += 1


        game_state[U.P1][U.TOKEN_STATES] = p1_state
        game_state[U.P1][U.SCORE] = 0 #Score track based on goal sector and fuel points
        game_state[U.P1][U.FUEL_SCORE] = 0 #Score track based on fuel remaining
        game_state[U.P2][U.TOKEN_STATES] = p2_state
        game_state[U.P2][U.SCORE] = 0 #Score track based on goal sector and fuel points
        game_state[U.P2][U.FUEL_SCORE] = 0 #Score track based on fuel remaining
        game_state[U.TURN_COUNT] = 0
        game_state[U.GAME_DONE] = False
        game_state[U.TURN_PHASE] = U.MOVEMENT

        # update token adjacency graph
        game_state[U.TOKEN_ADJACENCY] = get_token_adjacency_graph(self.board_grid, token_catalog)

        # update legal actions
        game_state[U.LEGAL_ACTIONS] = get_legal_verbose_actions(
            turn_phase=game_state[U.TURN_PHASE],
            token_catalog=token_catalog,
            board_grid=self.board_grid,
            token_adjacency_graph=game_state[U.TOKEN_ADJACENCY],
            min_ring=self.inargs.min_ring,
            max_ring=self.inargs.max_ring)

        return game_state, token_catalog, n_tokens_alpha, n_tokens_beta
    
    def arbitrary_game_state_from_server(self,cur_game_state) -> Tuple:
        import orbit_defender2d.king_of_the_hill.game_server as GS
        ''' returns kothgame formatted game_state based on gameserver information. 
            Used to keep the KothGame functions up to date with the game server.
            Requires importing the GS, which in turn imports KothGame, therefore
            could cause circular import issues if not careful.
        Args:
            cur_game_state (Dict): state of game passed from the game server
            
        Returns:
            game_state (Dict): state of game and tokens within game
            token_catalog (Dict): token states with token names as keys
            n_token_alpha (int): number of tokens for player alpha
            n_token_beta (int): number of tokens for player beta 
        '''
        game_state = {U.P1:dict(), U.P2:dict()}
        token_catalog = OrderedDict()
        
        # Specify goal locations (i.e. "hills") in geo 180 degrees offset
        game_state[U.GOAL1] = cur_game_state[GS.GOAL_ALPHA] #p1hill
        game_state[U.GOAL2] = cur_game_state[GS.GOAL_BETA] #p2hill

        # Populate the seeker pieces at team target sectors (hills)
        p1_state = [None] #TODO: Get rid of hardcoded piece names
        p1_state[0] = token_catalog["alpha:seeker:0"] = KOTHTokenState(
                Satellite(fuel=cur_game_state[GS.TOKEN_STATES][0]['fuel'], ammo=cur_game_state[GS.TOKEN_STATES][0]['ammo']), 
                role=cur_game_state[GS.TOKEN_STATES][0]['role'], 
                position=cur_game_state[GS.TOKEN_STATES][0]['position'])
        n_tokens_alpha = 1

        p2_state = [None]
        p2_state[0] = token_catalog["beta:seeker:0"] = KOTHTokenState(
                Satellite(fuel=cur_game_state[GS.TOKEN_STATES][1]['fuel'], ammo=cur_game_state[GS.TOKEN_STATES][1]['ammo']), 
                role=cur_game_state[GS.TOKEN_STATES][1]['role'], 
                position=cur_game_state[GS.TOKEN_STATES][1]['position'])
        n_tokens_beta = 1

        # Populate team bludger pieces based on init_pattern relative to target sectors (hills)
        n_sats = len(cur_game_state[GS.TOKEN_STATES])/2 - 1
        if n_sats % 1 != 0:
            raise ValueError("Uneven number of satellites")
        else:
            n_sats = int(n_sats)

        for sat_i in range(n_sats):
            p1_state.append(None)
            p1_state[-1] = token_catalog[U.P1 + U.TOKEN_DELIMITER + U.BLUDGER + U.TOKEN_DELIMITER + str(n_tokens_alpha)] = \
                KOTHTokenState(
                    Satellite(fuel=cur_game_state[GS.TOKEN_STATES][n_tokens_alpha+1]['fuel'], ammo=cur_game_state[GS.TOKEN_STATES][n_tokens_alpha+1]['ammo']), 
                    role=cur_game_state[GS.TOKEN_STATES][n_tokens_alpha+1]['role'], 
                    position=cur_game_state[GS.TOKEN_STATES][n_tokens_alpha+1]['position'])
            n_tokens_alpha += 1

        
        for sat_i in range(n_sats):
            p2_state.append(None)
            p2_state[-1] = token_catalog[U.P2 + U.TOKEN_DELIMITER + U.BLUDGER + U.TOKEN_DELIMITER + str(n_tokens_beta)] = \
                KOTHTokenState(
                    Satellite(fuel=cur_game_state[GS.TOKEN_STATES][n_tokens_alpha+n_tokens_beta]['fuel'], ammo=cur_game_state[GS.TOKEN_STATES][n_tokens_alpha+n_tokens_beta]['ammo']), 
                    role=cur_game_state[GS.TOKEN_STATES][n_tokens_alpha+n_tokens_beta]['role'], 
                    position=cur_game_state[GS.TOKEN_STATES][n_tokens_alpha+n_tokens_beta]['position'])
            n_tokens_beta += 1

        game_state[U.P1][U.TOKEN_STATES] = p1_state #p1state
        game_state[U.P1][U.SCORE] = cur_game_state[GS.SCORE_ALPHA] #p1score
        game_state[U.P2][U.TOKEN_STATES] = p2_state #p2state
        game_state[U.P2][U.SCORE] = cur_game_state[GS.SCORE_BETA] #p2score
        game_state[U.TURN_COUNT] = cur_game_state[GS.TURN_NUMBER] #turn number
        game_state[U.GAME_DONE] = cur_game_state[GS.GAME_DONE] #game done
        game_state[U.TURN_PHASE] = cur_game_state[GS.TURN_PHASE] #turn phase

        # update token adjacency graph
        game_state[U.TOKEN_ADJACENCY] = get_token_adjacency_graph(self.board_grid, token_catalog)

        # update legal actions
        game_state[U.LEGAL_ACTIONS] = get_legal_verbose_actions(
            turn_phase=game_state[U.TURN_PHASE],
            token_catalog=token_catalog,
            board_grid=self.board_grid,
            token_adjacency_graph=game_state[U.TOKEN_ADJACENCY],
            min_ring=self.inargs.min_ring,
            max_ring=self.inargs.max_ring)

        return game_state, token_catalog, n_tokens_alpha, n_tokens_beta

    def update_turn_phase(self, turn_phase: str):
        ''' update game_state with new turn phase; updates adjacency and legal acitons
        '''
        assert turn_phase in U.TURN_PHASE_LIST
        
        # update turn phase
        self.game_state[U.TURN_PHASE] = turn_phase
        # update adjacency THEN update legal actions
        self.update_token_adjacency_graph()

        # update legal actions
        self.update_legal_verbose_actions()

    def get_engagement_probability(self, token_id: str, target_id: str, engagement_type: str) -> float:
        ''' return probability of engagement success based on adjacency graph
        
        Args:
            token_id (str): identifier for instagating token
            target_id (str): identifier for targetted token
            engagement_type (str): identifier for engagement type
        
        Returns:
            prob (float): probability of engagement success
        '''
         # Make sure this function is always called when creating engagement touples that will be passed to resolve_engagements

        prob = 0.0
        if token_id.split(U.TOKEN_DELIMITER)[0] == U.P1:
        # check if adjacent, return 0 otherwise
            if engagement_type == U.NOOP:
                prob = self.inargs.engage_probs[U.P1][U.IN_SEC][U.NOOP]
            elif self.game_state[U.TOKEN_ADJACENCY].has_edge(token_id,target_id):
                if self.token_catalog[token_id].position == self.token_catalog[target_id].position:
                    prob = self.inargs.engage_probs[U.P1][U.IN_SEC][engagement_type]
                else:
                    prob = self.inargs.engage_probs[U.P1][U.ADJ_SEC][engagement_type]
            return prob
        else:
            if engagement_type == U.NOOP:
                prob = self.inargs.engage_probs[U.P2][U.IN_SEC][U.NOOP]
            elif self.game_state[U.TOKEN_ADJACENCY].has_edge(token_id,target_id):
                if self.token_catalog[token_id].position == self.token_catalog[target_id].position:
                    prob = self.inargs.engage_probs[U.P2][U.IN_SEC][engagement_type]
                else:
                    prob = self.inargs.engage_probs[U.P2][U.ADJ_SEC][engagement_type]
            return prob

    def update_token_adjacency_graph(self):
        self.game_state[U.TOKEN_ADJACENCY] = get_token_adjacency_graph(self.board_grid, self.token_catalog)

    def update_legal_verbose_actions(self):
        self.game_state[U.LEGAL_ACTIONS] = get_legal_verbose_actions(
            turn_phase=self.game_state[U.TURN_PHASE],
            token_catalog=self.token_catalog,
            board_grid=self.board_grid,
            token_adjacency_graph=self.game_state[U.TOKEN_ADJACENCY],
            min_ring=self.inargs.min_ring,
            max_ring=self.inargs.max_ring)

    def enforce_legal_verbose_actions(self, actions: Dict):
        ''' ensure all actions are legal, terminate game if not

        Args:
            actions (dict): verbose action description
                            key is piece id token_catalog, one for each piece in game
                            value is the piece's movement tuple (U.MOVEMENT_TYPES) or 
                            engagement tuple (ENGAGMENT_TYPE, target_piece_id, prob)

        Returns:
            True if all actions are valid

        '''
        illegal_actions, alpha_illegal, beta_illegal = get_illegal_verbose_actions(actions, self.game_state[U.LEGAL_ACTIONS])
        if len(illegal_actions) > 0:
            # set player's score if illegal action selected
            if alpha_illegal:
                self.game_state[U.P1][U.SCORE] = self.inargs.illegal_action_score
            if beta_illegal:
                self.game_state[U.P2][U.SCORE] = self.inargs.illegal_action_score

            # end game
            return False

        else:
            return True

    def apply_verbose_actions(self, actions: Dict):
        ''' apply move or engagement actions to update game state.

        The actions are encoded in verbose format (as opposed to gym format)
        This is the closest corellary to step() in gym api

        Note that only rewards from current timestep are returned. Game state
        (and therefore observation of game state) is updated internally but
        not returned explicitly

        Args:
            actions (dict): verbose action description
                            key is piece id token_catalog, one for each piece in game
                            value is the piece's movement tuple (U.MOVEMENT_TYPES) or 
                            engagement tuple (ENGAGMENT_TYPE, target_piece_id, prob)
            render_active (bool): tells function whether to return engagement outcomes
                            for rendering

        Returns:
            rewards : dict
                rewards earned by each player during time step
                These are the rewards that pettingzoo envs uses to help train the AI agents. They are zero unless the game ends... which is not great...

        '''

        if self.game_state[U.TURN_PHASE] in [U.MOVEMENT, U.ENGAGEMENT]:

            # ensure all actions are legal, terminate game if not
            all_valid_acts = self.enforce_legal_verbose_actions(actions)

            # evaluate
            if not all_valid_acts:
                # If there are illegal actions, then terminate the game. Illegal action score is applied in enforce_legal_verbose_actions so doesn't need to be updated here.
                return self.terminate_game()

            # decrement fuel of pieces based on legal action selection, 
            # ammend actions to remove those that have insufficient fuel
            fuel_constrained_actions = self.apply_fuel_constraints(actions)

            if self.game_state[U.TURN_PHASE] == U.MOVEMENT:
                # apply movement
                self.move_pieces(fuel_constrained_actions)

                # update turn phase, adjacency graph, and legal actions
                self.update_turn_phase(U.ENGAGEMENT)
            
            else:
                # apply engagement actions
                self.engagement_outcomes = self.resolve_engagements(engagements=fuel_constrained_actions)
                self.enact_engagements()

                # update turn phase, adjacency graph, and legal actions
                self.update_turn_phase(U.DRIFT)

        elif self.game_state[U.TURN_PHASE] == U.DRIFT:
            
            # no actions to apply in drift, move pieces one sector prograde
            assert actions is None

            #Get fuel points for each player for this turn
            alpha_fuel_points = self.get_fuel_points(player_id=U.P1)
            beta_fuel_points = self.get_fuel_points(player_id=U.P2)

            #Get teh goal sector points for each player
            alpha_goal_points, beta_goal_points = self.get_points()

            #Get the points for each player from the last round and subtract the fuel score from the last round to get just the goal points (which are cumulative)
            alpha_score_just_goal = self.game_state[U.P1][U.SCORE] - self.game_state[U.P1][U.FUEL_SCORE]
            beta_score_just_goal = self.game_state[U.P2][U.SCORE] - self.game_state[U.P2][U.FUEL_SCORE]

            #Add this turn's goal points to the cumulative goal points
            alpha_score_just_goal += alpha_goal_points
            beta_score_just_goal += beta_goal_points

            #Add this turn's fuel points to the cumulative goal points to update game score
            self.game_state[U.P1][U.SCORE] = alpha_score_just_goal + alpha_fuel_points
            self.game_state[U.P2][U.SCORE] = beta_score_just_goal + beta_fuel_points

            #Update the fuel score for each player
            self.game_state[U.P1][U.FUEL_SCORE] = alpha_fuel_points
            self.game_state[U.P2][U.FUEL_SCORE] = beta_fuel_points

            # evaluate game termination conditions
            if self.is_terminal_game_state():
                return self.terminate_game()

            # decrement fuel and move pieces
            for token_name, token_state in self.token_catalog.items():
                if token_name.split(U.TOKEN_DELIMITER)[0] == U.P1:
                    # decrement station keeping fuel
                    token_state.satellite.fuel -= self.inargs.fuel_usage[U.P1][U.DRIFT]
                    token_state.satellite.fuel = max(token_state.satellite.fuel, self.inargs.min_fuel)
                    # move tokens one sector prograde
                    token_state.position = self.board_grid.get_prograde_sector(token_state.position)
                else:
                    # decrement station keeping fuel
                    token_state.satellite.fuel -= self.inargs.fuel_usage[U.P2][U.DRIFT]
                    token_state.satellite.fuel = max(token_state.satellite.fuel, self.inargs.min_fuel)
                    # move tokens one sector prograde
                    token_state.position = self.board_grid.get_prograde_sector(token_state.position)

            # move goal sectors one sector prograde
            self.game_state[U.GOAL1] = self.board_grid.get_prograde_sector(self.game_state[U.GOAL1])
            self.game_state[U.GOAL2] = self.board_grid.get_prograde_sector(self.game_state[U.GOAL2])

            # increment turn counter
            self.game_state[U.TURN_COUNT] += 1

            # update turn phase, adjacency graph, and legal actions
            self.update_turn_phase(U.MOVEMENT)

        else:
            raise ValueError("Unrecognized game phase {}".format(self.game_state[U.TURN_PHASE]))

        # if non-terminal game, return zero rewards
        return {plr_id:0.0 for plr_id in self.player_names}

    def apply_fuel_constraints(self, actions: Dict) -> Dict:
        ''' decrement fuel in game_state based on actions and return fuel-constrained actions

        Args:
            actions (dict): key is piece id, one for each piece in game
                            value is the piece's action tuple
        
        Returns:
            fuel_constrained_actions (dict): actions ammended based on fuel constraints
        '''
        
        fuel_constrained_actions = deepcopy(actions) 
        for token_name, action_tuple in actions.items():
            if token_name.split(U.TOKEN_DELIMITER)[0] == U.P1:
                fuel = self.token_catalog[token_name].satellite.fuel

                # determine fuel needed for action
                fuel_usage = None
                min_fuel_action_tuple = None
                if action_tuple.action_type in U.MOVEMENT_TYPES:
                    # movement fuel usage independent of sector and target
                    fuel_usage = self.inargs.fuel_usage[U.P1][action_tuple.action_type]
                    min_fuel_action_tuple = U.MovementTuple(U.NOOP)
                elif action_tuple.action_type in U.ENGAGEMENT_TYPES:
                    min_fuel_action_tuple = U.EngagementTuple(U.NOOP, token_name, None)
                    target_name = action_tuple.target
                    if self.token_catalog[token_name].position  == self.token_catalog[target_name].position:
                        fuel_usage = self.inargs.fuel_usage[U.P1][U.IN_SEC][action_tuple.action_type]
                    elif target_name in self.game_state[U.TOKEN_ADJACENCY].neighbors(token_name):
                        fuel_usage = self.inargs.fuel_usage[U.P1][U.ADJ_SEC][action_tuple.action_type]
                    else:
                        raise ValueError("Invalid engagement {} between {} and {}".format(
                            action_tuple.action_type,
                            token_name,
                            target_name))

                else:
                    raise ValueError("Unrecognized action type {}".format(action_tuple.action_type))

                fuel -= fuel_usage
                if fuel < self.inargs.min_fuel:
                    # insufficient fuel, leave fuel as is and
                    # set action to noop movement
                    #self.token_catalog[token_name].satellite.fuel = self.inargs.min_fuel #This was to take all remaining fuel away then still set action to noop. I will instead leave fuel as is and stop the action from happening.
                    # If the fuel begins as zero, then it will always be below min_fuel and the action will always be noop. This is used for tokens remvoved from the game.
                    fuel_constrained_actions[token_name] = min_fuel_action_tuple
                else:
                    # sufficient fuel, decrement fuel and copy action
                    self.token_catalog[token_name].satellite.fuel = fuel
                    fuel_constrained_actions[token_name] = action_tuple
            else:
                fuel = self.token_catalog[token_name].satellite.fuel

                # determine fuel needed for action
                fuel_usage = None
                min_fuel_action_tuple = None
                if action_tuple.action_type in U.MOVEMENT_TYPES:
                    # movement fuel usage independent of sector and target
                    fuel_usage = self.inargs.fuel_usage[U.P2][action_tuple.action_type]
                    min_fuel_action_tuple = U.MovementTuple(U.NOOP)
                elif action_tuple.action_type in U.ENGAGEMENT_TYPES:
                    min_fuel_action_tuple = U.EngagementTuple(U.NOOP, token_name, None)
                    target_name = action_tuple.target
                    if self.token_catalog[token_name].position  == self.token_catalog[target_name].position:
                        fuel_usage = self.inargs.fuel_usage[U.P2][U.IN_SEC][action_tuple.action_type]
                    elif target_name in self.game_state[U.TOKEN_ADJACENCY].neighbors(token_name):
                        fuel_usage = self.inargs.fuel_usage[U.P2][U.ADJ_SEC][action_tuple.action_type]
                    else:
                        raise ValueError("Invalid engagement {} between {} and {}".format(
                            action_tuple.action_type,
                            token_name,
                            target_name))

                else:
                    raise ValueError("Unrecognized action type {}".format(action_tuple.action_type))

                fuel -= fuel_usage
                if fuel < self.inargs.min_fuel:
                    # insufficient fuel, leave fuel as is and
                    # set action to noop movement
                    #self.token_catalog[token_name].satellite.fuel = self.inargs.min_fuel #This was to take all remaining fuel away then still set action to noop. I will instead leave fuel as is and stop the action from happening.
                    # If the fuel begins as zero, then it will always be below min_fuel and the action will always be noop. This is used for tokens remvoved from the game.
                    fuel_constrained_actions[token_name] = min_fuel_action_tuple
                else:
                    # sufficient fuel, decrement fuel and copy action
                    self.token_catalog[token_name].satellite.fuel = fuel
                    fuel_constrained_actions[token_name] = action_tuple

        return fuel_constrained_actions

    def resolve_engagements(self, engagements: Dict):
        ''' determine the outcome of attack and defense actions

        Args:
            engagements (dict): key is piece id, one for each piece in game
                                value is the piece's EngagementTuple (engagement_type, target_piece_id, prob)
        
        Returns:
            eg_outcomes (list): sequence of EngagementOutcomeTuple
        '''

        # create sequential list of engagement outcomes
        eg_outcomes = []

        # create initial engagement graph, passing the game state of each players' pieces
        eg = EngagementGraph(engagements)

        # prune zero-fuel piece nodes
        zero_fuel_pieces = [k for k in self.token_catalog.keys() if self.token_catalog[k].satellite.fuel <= self.inargs.min_fuel]
        eg.egraph.remove_nodes_from(zero_fuel_pieces)

        # Evaluate guard action success, 
        # rerouting attacks to guardian pieces for successful guards
        # TODO: check legality of guard engagements against legal action mask (e.g. guarding non-adjacent sectors)
        eg_outcomes.extend(eg.resolve_guard_engagements())

        # simultaneously resolve all shoot attacks, 
        # removing nodes for successful attacks and edges for unsuccessful
        # TODO: check legality of shoot engagements against legal action mask (e.g. single-use kinetic attack, shooting non-adjacent sectors) 
        eg_outcomes.extend(eg.resolve_shoot_engagements())

        # resolve all one-way collision attacks in random order
        # TODO: check legality of collide engagements against legal action mask (e.g. colliding distant pieces) 
        eg_outcomes.extend(eg.resolve_collide_engagements())

        # check for complete engagement graph resolution
        assert eg.egraph.number_of_edges() == 0, \
            "Unsuccessful engagement resolution with {} edges remaining".format(eg.egraph.number_of_edges)
        
        return eg_outcomes

    def enact_engagements(self) -> None:
        ''' update game state based on current engagement outcomes
        '''

        for egout in self.engagement_outcomes:

            if egout.action_type == U.SHOOT:    # handle shoot engagement outcomes
                # remove shoot capability from pieces that expend shot
                self.token_catalog[egout.attacker].satellite.ammo -= 1
                
                # zero-out fuel of destroyed target token if engagement successful
                if egout.success:
                    self.token_catalog[egout.target].satellite.fuel = self.inargs.min_fuel

            elif egout.action_type == U.COLLIDE:    # handle collide engagement outcomes

                # move pieces to target sector if they chose an adjacent-sector engagement
                self.token_catalog[egout.attacker].position = self.token_catalog[egout.target].position

                # zero-out fuel of destroyed attacker and target token if engagement successful
                if egout.success:
                    self.token_catalog[egout.attacker].satellite.fuel = self.inargs.min_fuel
                    self.token_catalog[egout.target].satellite.fuel = self.inargs.min_fuel
            
            elif egout.action_type == U.GUARD: # handle guard engagement outcomes

                # move pieces to target sector if they chose and adjacent-sector engagement
                self.token_catalog[egout.guardian].position = self.token_catalog[egout.target].position

            elif egout.action_type == U.NOOP:
                pass

            else:
                raise ValueError("Unrecognized action_type {}".format(egout.action_type))


    def move_pieces(self, moves: Dict) -> None:
        ''' move each piece on the board

        Args:
            moves (dict): key is piece id, one for each piece in game
                            value is the piece's MovementTuple (movement_type)
        '''
        for piece_name, move_tup in moves.items():
            
            # get current position (sector) of piece
            cur_sec = self.token_catalog[piece_name].position

            move = move_tup.action_type
            if move == U.NOOP:
                pass
            elif move == U.PROGRADE:
                self.token_catalog[piece_name].position = self.board_grid.get_prograde_sector(cur_sec)
            elif move == U.RETROGRADE:
                self.token_catalog[piece_name].position = self.board_grid.get_retrograde_sector(cur_sec)
            elif move == U.RADIAL_IN:
                self.token_catalog[piece_name].position = self.board_grid.get_radial_in_sector(cur_sec)
            elif move == U.RADIAL_OUT:
                self.token_catalog[piece_name].position = self.board_grid.get_radial_out_sector(cur_sec)
            else:
                raise ValueError("Unrecognized piece movement {} for piece {}".format(move, piece_name))

    def get_points(self) -> Tuple:
        ''' evaluate the points scored based on current game state

        Args:
            None

        Returns:
            alpha_points (float): points scored by alpha player
            beta_points (float): points scored by beta player
        '''
        alpha_goal_points_sec = self.inargs.in_goal_points[U.P1]
        alpha_goal_points_adj_secs = self.inargs.adj_goal_points[U.P1]
        beta_goal_points_sec = self.inargs.in_goal_points[U.P2]
        beta_goal_points_adj_secs = self.inargs.adj_goal_points[U.P2]

        # evaluate goal-adjacent sectors
        alpha_goal_sec = self.game_state[U.GOAL1]
        alpha_goal_adj_secs = self.board_grid.get_all_adjacent_sectors(alpha_goal_sec)
        beta_goal_sec = self.game_state[U.GOAL2]
        beta_goal_adj_secs = self.board_grid.get_all_adjacent_sectors(beta_goal_sec)

        # get current location of all seeker tokens
        alpha_seeker_secs = [tok.position for tok_name, tok in self.token_catalog.items() 
                            if U.P1 + U.TOKEN_DELIMITER + U.SEEKER in tok_name]
        beta_seeker_secs = [tok.position for tok_name, tok in self.token_catalog.items() 
                            if U.P2 + U.TOKEN_DELIMITER + U.SEEKER in tok_name]

        # evaluate points scored
        alpha_points = 0
        beta_points = 0
        for aseek in alpha_seeker_secs:
            if aseek == alpha_goal_sec:
                alpha_points += alpha_goal_points_sec
            elif aseek in alpha_goal_adj_secs:
                alpha_points += alpha_goal_points_adj_secs
        for bseek in beta_seeker_secs:
            if bseek == beta_goal_sec:
                beta_points += beta_goal_points_sec
            elif bseek in beta_goal_adj_secs:
                beta_points += beta_goal_points_adj_secs

        return alpha_points, beta_points

    def is_terminal_game_state(self) -> bool:
        ''' check for terminal game state conditions
        
        Args:
            None

        Returns:
            is_terminal (bool): true if terminal game state
        '''
        is_terminal = False

        if (self.game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel <= self.inargs.min_fuel or
            self.game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel <= self.inargs.min_fuel or
            self.game_state[U.P1][U.SCORE] >= self.inargs.win_score[U.P1] or 
            self.game_state[U.P2][U.SCORE] >= self.inargs.win_score[U.P2] or
            self.game_state[U.TURN_COUNT] >= self.inargs.max_turns):
            is_terminal = True

        return is_terminal

    # def get_fuel_points_old(self, player_id):
    #     '''convert fuel remaining in seeker tokens to points'''
    #     seeker_tok = self.get_token_id(player_id=player_id, token_num=0)
    #     assert U.SEEKER in seeker_tok
    #     return self.token_catalog[seeker_tok].satellite.fuel * self.inargs.fuel_points_factor

    def get_fuel_points(self, player_id):
        '''convert fuel remaining in all tokens to points'''
        fuel_points = 0
        for token_name, token_state in self.token_catalog.items():
            if token_name.startswith(player_id):
                #if token is a seeker then add the fuel points to the total
                if token_state.role == U.SEEKER:
                    if token_state.satellite.fuel > 0:
                        if player_id == U.P1:
                            fuel_points += token_state.satellite.fuel * self.inargs.fuel_points_factor[U.P1]
                        else:
                            fuel_points += token_state.satellite.fuel * self.inargs.fuel_points_factor[U.P2]
                #if token is a bludger then add the fuel points to the total with fuel_points_bludger_factor (hard code as 0.1 for now should add this to inargs later)
                elif token_state.role == U.BLUDGER:
                    if token_state.satellite.fuel > 0:
                        if player_id == U.P1:
                            fuel_points += token_state.satellite.fuel * self.inargs.fuel_points_factor_bludger[U.P1]
                        else:
                            fuel_points += token_state.satellite.fuel * self.inargs.fuel_points_factor_bludger[U.P2]
        return int(np.floor(fuel_points))

    def get_random_valid_actions(self) -> Dict:
        '''create a random-yet-valid action for each token
        
        Returns:
            actions (dict): verbose action description
                            key is piece id token_catalog, one for each piece in game
                            value is the piece's movement tuple (U.MOVEMENT_TYPES) or 
                            engagement tuple (ENGAGMENT_TYPE, target_piece_id, prob)
        '''
        actions = None
        if self.game_state[U.TURN_PHASE] != U.DRIFT:
            actions = {t:a[np.random.choice(len(a))] for t, a in self.game_state[U.LEGAL_ACTIONS].items()}

            # apply appropriate probabilities for engagements
            if self.game_state[U.TURN_PHASE] == U.ENGAGEMENT:
                actions = {t:U.EngagementTuple(
                    action_type=a.action_type, 
                    target=a.target, 
                    prob=self.get_engagement_probability(t, a.target, a.action_type)) for  t, a in actions.items()}
            
        return actions
    
    def get_noop_actions(self) -> Dict:
        '''get only noop actions for each token 
            Useful for debugging when you want an inactive player
        
        Returns:
            actions (dict): verbose action description
                            key is piece id token_catalog, one for each piece in game
                            value is the piece's movement tuple (U.MOVEMENT_TYPES) or 
                            engagement tuple (ENGAGMENT_TYPE, target_piece_id, prob)
        '''
        actions = None
        if self.game_state[U.TURN_PHASE] != U.DRIFT:
            actions = {t:a[0] for t, a in self.game_state[U.LEGAL_ACTIONS].items()}

            # apply appropriate probabilities for engagements
            if self.game_state[U.TURN_PHASE] == U.ENGAGEMENT:
                actions = {t:U.EngagementTuple(
                    action_type=a.action_type, 
                    target=a.target, 
                    prob=self.get_engagement_probability(t, a.target, a.action_type)) for  t, a in actions.items()}
            
        return actions

    def get_token_id(self, player_id, token_num):
        '''get full token id from player name and token number'''
        tok_id = [tid for tid in self.token_catalog.keys() if (
            tid.split(U.TOKEN_DELIMITER)[0] == player_id and
            tid.split(U.TOKEN_DELIMITER)[2] == str(token_num))]
        if len(tok_id) != 1:
            raise ValueError("Unexpected number of valid token IDs. Expected single id, got {}".format(tok_id))
        return tok_id[0]
    
    def get_input_actions(self, plr_id=U.P2):
        '''
        Get the actions from the human player as command line input from the terminal and format as a dictionary to pass to game server

        Input: User action selections from terminal

        Output: Dictionary of verbose actions to pass to KothGame and/or GameServer
        '''
        if plr_id == U.P2:
            target_plr_id = U.P1
        elif plr_id == U.P1:
            target_plr_id = U.P2

        actions_dict = {}
        movements = [U.NOOP, U.PROGRADE, U.RETROGRADE, U.RADIAL_IN, U.RADIAL_OUT]
        engagements = [U.NOOP, U.SHOOT, U.COLLIDE, U.GUARD]
        if self.game_state[U.TURN_PHASE] == U.MOVEMENT:
            for token_name, token_state in self.token_catalog.items():
                if token_state.satellite.fuel > self.inargs.min_fuel:
                    if token_name.startswith(plr_id):
                        #clear the screen
                        print("\n"*5)
                        print("Turnphase: {}".format(self.game_state[U.TURN_PHASE]))
                        print("Token: {}".format(token_name))
                        print("Select an action from the list")
                        print("0 - NOOP \n 1 - Prograde \n 2 - Retrograde \n 3 - Radial In \n 4 - Radial Out")
                        select_valid = 0
                        while not select_valid:
                            selection = input("Select action: ")
                            if selection.isdigit() and int(selection) < len(movements):
                                    if U.MovementTuple(movements[int(selection)]) in self.game_state[U.LEGAL_ACTIONS][token_name]:
                                        select_valid = 1
                                    else:
                                        print("Invalid selection. Please select a legal action")
                            else:
                                print("Invalid selection. Please select a number between 0 and {}".format(len(movements)-1))
                        #add the action to the dictionary of actions to send to the game server
                        actions_dict[token_name] = U.MovementTuple(movements[int(selection)])
                    else:
                        pass #don't do anything for actions for the other player
                else:
                    actions_dict[token_name] = U.MovementTuple(U.NOOP) #If the token is out of fuel then it can't move
        elif self.game_state[U.TURN_PHASE] == U.ENGAGEMENT:
            for token_name, token_state in self.token_catalog.items():
                if token_state.satellite.fuel > self.inargs.min_fuel:
                    action_valid = 0
                    if token_name.startswith(plr_id):
                        action_valid = 0
                        print("\n"*5)
                        while not action_valid:
                            #clear the screen
                            print("Turnphase: {}".format(self.game_state[U.TURN_PHASE]))
                            print("Token ID: {}".format(token_name))
                            print("Select an action from the list")
                            print("0 - NOOP \n 1 - Shoot \n 2 - Collide \n 3 - Guard")
                            select_valid = 0
                            while not select_valid:
                                selection = input("Select action: ")
                                if selection.isdigit() and int(selection) < len(engagements):
                                        select_valid = 1
                                else:
                                        print("Invalid selection. Please select a number between 0 and {}".format(len(engagements)-1))
                            #if the action is not a noop then prompt the player to select a target
                            if int(selection) != 0:
                                tgt_valid = 0
                                while not tgt_valid:
                                    tgt = input("Select target: ")
                                    if tgt.isdigit():
                                        if int(tgt) < 11:
                                            tgt_valid = 1
                                        else:
                                            print("Invalid selection. Please select a number between 0 and {}".format(10))
                                    else:
                                            print("Invalid selection. Please select a number between 0 and {}".format(10))
                            else:
                                tgt = 0    
                            #add the action to the dictionary of actions to send to the game server
                            #For engagement phase, the legal actions are a list with entries of actionType and then targetID
                            if int(selection) >0 and int(selection) < 3:
                                if int(tgt) != 0: #The target value will be 0 for the seeker and a number between 1 and 10 for the bludgers
                                    actions_dict[token_name] = U.EngagementTuple(engagements[int(selection)], \
                                        target_plr_id + U.TOKEN_DELIMITER + U.BLUDGER + U.TOKEN_DELIMITER + tgt, \
                                        self.get_engagement_probability(token_id=token_name,target_id=target_plr_id \
                                        + U.TOKEN_DELIMITER + U.BLUDGER + U.TOKEN_DELIMITER + tgt,engagement_type=engagements[int(selection)]))
                                else:
                                    actions_dict[token_name] = U.EngagementTuple(engagements[int(selection)], \
                                        target_plr_id + U.TOKEN_DELIMITER + U.SEEKER + U.TOKEN_DELIMITER + tgt, \
                                        self.get_engagement_probability(token_id=token_name,target_id=target_plr_id \
                                        + U.TOKEN_DELIMITER + U.SEEKER + U.TOKEN_DELIMITER + tgt,engagement_type=engagements[int(selection)]))
                            elif int(selection) == 3:
                                # If selection is 3, then the action is guard. The target will have the same player ID as the token that is guarding
                                actions_dict[token_name] = U.EngagementTuple(engagements[int(selection)], \
                                    plr_id + U.TOKEN_DELIMITER + U.SEEKER + U.TOKEN_DELIMITER + tgt, \
                                    self.get_engagement_probability(token_id=token_name,target_id=target_plr_id \
                                    + U.TOKEN_DELIMITER + U.SEEKER + U.TOKEN_DELIMITER + tgt,engagement_type=engagements[int(selection)]))
                            elif int(selection) == 0: #If the selection is 0, then the action is NOOP
                                actions_dict[token_name] = U.EngagementTuple(engagements[int(selection)], token_name,None)
                            else:
                                print("Unkonwn action type")
                            #Check that the action is legal is_legal_verbose_action(token, action, legal_actions):
                            if not is_legal_verbose_action(token_name, actions_dict[token_name], self.game_state[U.LEGAL_ACTIONS]):
                                print("Invalid selection. Please select a legal action")
                            else:
                                action_valid = 1
                    else:
                        pass #don't do anything for actions for the other player
                else:
                    actions_dict[token_name] = U.EngagementTuple(U.NOOP, token_name,None) #If the token is out of fuel then it can't take an action
        else:
            #for DRFIT phase this function won't be called
            raise ValueError("Invalid turn phase")

        #Check that none of the actions are illegal
        all_valid_acts = self.enforce_legal_verbose_actions(actions_dict)

        # evaluate
        if not all_valid_acts:
            print("Illegal actions selected, please try again")
            actions_dict = self.get_input_actions(plr_id=plr_id)
        return actions_dict

def parse_token_id(t):
    ''' get player_id, role, and token_num from token_id
    
    Args:
        t (str): token identifier string

    Returns:
        (player_id, role, token_num) 
    '''
    tsplit = t.split(U.TOKEN_DELIMITER)
    return tsplit[0], tsplit[1], tsplit[2]

def is_same_player(t1, t2):
    ''' check if tokens are from same player
    
    Args:
        t1 (str): name of token 1
        t2 (str): name of token 2
    
    Returns:
        (bool): true if tokens are from same player
    '''
    p1, _, _ = parse_token_id(t1)
    p2, _, _ = parse_token_id(t2)
    if p1 == p2:
        return True
    else:
        return False

def get_token_adjacency_graph(board, token_catalog):
    ''' create graph with tokens as nodes and edges between tokens that are in adjacent sectors

    Args:
        board (OrbitGrid): game board
        token_catalog (Dict): key is game token id (one entry for each game token), value is game token state

    Returns:
        adj_graph (DiGraph): networkx directed graph describing adjacency of tokens
    '''

    # Instantiate directional engagement graph
    adj_graph = nx.DiGraph()

    # create nodes from game piece objects
    adj_graph.add_nodes_from(token_catalog.keys())

    # populate edges based on token locations
    for token1_name, token1_state in token_catalog.items():
        
        # get sectors movement-adjacent to token1
        # also include both radial out sectors, if applicable
        token1_sec = token1_state.position
        adj_sec = []
        adj_sec.append(token1_sec)
        adj_sec.extend(board.get_all_adjacent_sectors(token1_sec))

        for token2_name, token2_state in token_catalog.items():

            # Skip for same token
            if token1_name == token2_name:
                continue

            # check if token2 in adjacent sector
            if token2_state.position in adj_sec:
                adj_graph.add_edge(token1_name, token2_name)

    return adj_graph

def get_illegal_verbose_actions(actions: Dict, legal_actions: Dict):
    ''' return dictionary of illegal actions. Does not check legality of probability of engagement

    Args:
        actions (Dict): action dictionary to be checked for illegal actions
        legal_actions (Dict): dictionary of legal actions
    
    Returns:
        illegal_actions (Dict): illegal actions in actions dictionary
        alpha_illegal (bool): true if player alpha selected any illegal actions
        beta_illegal (bool): true if player beta selected any illegal actions
    '''
    #if actions is None return empty dict
    if actions is None:
        return dict(), False, False
    else:
        illegal_actions = {tok:act for tok,act in actions.items() if not is_legal_verbose_action(tok,act,legal_actions)}
        alpha_illegal = any([tok.split(U.TOKEN_DELIMITER)[0] == U.P1 for tok in illegal_actions.keys()])
        beta_illegal = any([tok.split(U.TOKEN_DELIMITER)[0] == U.P2 for tok in illegal_actions.keys()])
        return illegal_actions, alpha_illegal, beta_illegal
    

def is_legal_verbose_action(token, action, legal_actions):
    ''' check if single action is legal, ignoring probability

    Args:
        token (str): name of token
        action (namedTuple): MovementTuple or EngagementTuple
        legal_actions (dict): dictionary of legal actions
    
    Returns:
        (bool): true if action is legal
    '''

    if action == U.INVALID_ACTION:
        return False
    elif isinstance(action, U.MovementTuple):
        return action in legal_actions[token]
    elif isinstance(action, U.EngagementTuple):
        return U.EngagementTuple(action.action_type, action.target, None) in legal_actions[token]
    else:
        raise ValueError("Unrecognized action type {}".format(action))

def get_legal_verbose_actions(
    turn_phase: str, 
    token_catalog: Dict, 
    board_grid: OrbitGrid, 
    token_adjacency_graph: nx.DiGraph,
    min_ring: int, 
    max_ring: int):
    ''' given the current game state, determine which actions are currently legal for each piece

    Separated into non-member function because needs access to token_catalog before it is set as a member variable
    in reset

    Args:
        turn_phase (str): string description of game phase (movement, engagement, drift)
        token_catalog (Dict): token states with token names as keys
        board_grid (OrbitGrid): game board
        token_adjacency_graph (nx.DiGraph): graph with tokens as nodes and edges if tokens are adjacent in game board
        min_ring (int): innermost playable orbit ring in game board
        max_ring (int): outermost playable orbit ring in game board

    Returns:
        legal_actions (dict): key is piece name in catalog,
                                value is list of legal verbose actions for that piece
    '''

    legal_actions = dict()

    # iterate through each piece in the piece catalog
    for token_name, token_state in token_catalog.items():
        
        legal_actions[token_name] = []

        if turn_phase == U.MOVEMENT:
            if token_state.satellite.fuel <= 0:
                legal_actions[token_name].append(U.MovementTuple(U.NOOP))
            else:
                # no-operation, prograde, and retrograde always valid
                legal_actions[token_name].extend([
                    U.MovementTuple(U.NOOP), 
                    U.MovementTuple(U.PROGRADE), 
                    U.MovementTuple(U.RETROGRADE)])

                # radial_in valid if piece is not in min ring
                if board_grid.sector_num2ring(token_state.position) > min_ring:
                    legal_actions[token_name].append(U.MovementTuple(U.RADIAL_IN))

                # radial_out valid if piece is not in max ring
                if board_grid.sector_num2ring(token_state.position) < max_ring:
                    legal_actions[token_name].append(U.MovementTuple(U.RADIAL_OUT))

        elif turn_phase == U.ENGAGEMENT:
            if token_state.satellite.fuel <= 0:
                legal_actions[token_name].append(U.EngagementTuple(U.NOOP, token_name, None))
            else:
                # evaluate legal engagements for token

                # extract player name (it affects what actions are legal)
                player_name = token_name.split(U.TOKEN_DELIMITER)[0]

                # no-operation is always valid
                legal_actions[token_name].append(U.EngagementTuple(U.NOOP, token_name, None))

                # get valid engagements based on piece adjacency
                for target_token_name in token_adjacency_graph.neighbors(token_name):
                    assert target_token_name != token_name
                    target_player_name = target_token_name.split(U.TOKEN_DELIMITER)[0]

                    if player_name == target_player_name:
                        if U.SEEKER in target_token_name:
                            # guard is legal only for same player's seeker and only if at least one adjacent active token is not the same player as the player_name
                            if any([not is_same_player(token_adjacent_name, token_name) and token_catalog[token_adjacent_name].satellite.fuel > 0 for token_adjacent_name in token_adjacency_graph.neighbors(target_token_name)]):
                                legal_actions[token_name].append(U.EngagementTuple(U.GUARD, target_token_name, None))
                    else:
                        #Actions against the other player's tokens are only legal if the target token has fuel remaining (is not inactive)
                        if token_catalog[target_token_name].satellite.fuel > 0:
                            # collide is legal if target has fuel, even if the actor does not have enough fuel. If actor fuel is insufficient, then this will be filterd out by apply_fuel_constraints
                            legal_actions[token_name].append(U.EngagementTuple(U.COLLIDE, target_token_name, None))
                            # shoot only legal if ammo available
                            if token_state.satellite.ammo >= 1:
                                legal_actions[token_name].append(U.EngagementTuple(U.SHOOT, target_token_name, None))

        elif turn_phase == U.DRIFT:
            # no legal actions during drift
            pass

        else:
            raise ValueError("Unrecognized game phase {}".format(turn_phase))

    return legal_actions

def print_game_info(game, file=None):
    # print("alpha player state: ")
    # for tok in game.game_state[U.P1][U.TOKEN_STATES]:
    #     print("-->{} | fuel: {} | position: {}".format(tok.satellite.fuel, tok.position))
    print("STATES:", file=file)
    for toknm, tok in game.token_catalog.items():
        if tok.satellite.fuel >= 0 and tok.position > 0:
            print("   {:<16s}| position: {:<4d}| fuel: {:<8.1f} ".format(toknm, tok.position, tok.satellite.fuel), file=file)
    #print("alpha|beta score: {}|{}".format(game.game_state[U.P1][U.SCORE],game.game_state[U.P2][U.SCORE]))

def print_scores(game, file=None):
    #Print the turn number and the score for each player
    #print("Score at Turn : {} and Phase : {}".format(game.game_state[U.TURN_COUNT], game.game_state[U.TURN_PHASE]))
    print("alpha score: {}  |  beta score: {}".format(game.game_state[U.P1][U.SCORE],game.game_state[U.P2][U.SCORE]), file=file)

def print_actions(actions, file=None):
    print("ACTIONS:", file=file)
    if actions is None:
        print("   None", file=file)
    else:
        for toknm, act in actions.items():
            print("   {:<15s} | {}".format(toknm, act), file=file)

def print_engagement_outcomes(engagement_outcomes, file=None):
    print("ENGAGEMENT OUTCOMES:", file=file)
    # if engagement_outcomes is empty print No engagements
    if not engagement_outcomes:
        print("    No engagements", file=file)
    else:
        # print the engagement outcomes for guarding actions first
        print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format("Action", "Attacker", "Guardian", "Target", "Result"), file=file)
        for egout in engagement_outcomes:
            success_status = "Success" if egout.success else "Failure"
            if egout.action_type == U.SHOOT or egout.action_type == U.COLLIDE:
                print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format(
                    egout.action_type, egout.attacker, "", egout.target, success_status), file=file)
            elif egout.action_type == U.GUARD:
                if isinstance(egout.attacker, str):
                    print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format(
                        egout.action_type, egout.attacker, egout.guardian, egout.target, success_status), file=file)
                else:
                    print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format(
                        egout.action_type, "", egout.guardian, egout.target, success_status), file=file)
            elif egout.action_type == U.NOOP:
                print("NOOP", file=file)
            else:
                print("Unrecognized action type {}".format(egout.action_type), file=file)
                raise ValueError("Unrecognized action type {}".format(egout.action_type))
def start_log_file(logfile):
    ''' create a new game log file

    Args:
        logfile (str): path to game log file
    '''
    #Make new filename with date and time appended  to logfile
    logfile = logfile + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    #Check that the file exists and create it if it doesn't exist
    if not os.path.isfile(logfile):
        with open(logfile, 'w') as f:
            #Write header with date and time
            f.write("Game Log File\n")
            f.write("Date: {}\n".format(datetime.datetime.now()))
            f.close()

    return logfile

def print_endgame_status(game, file=None):
    '''
    Print the endgame status, scores, winner, termination conditions.
    '''
    winner = None
    alpha_score =game.game_state[U.P1][U.SCORE]
    beta_score = game.game_state[U.P2][U.SCORE]
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'

    cur_game_state = game.game_state
    if cur_game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel <= game.inargs.min_fuel:
        print("alpha seeker out of fuel", file=file)
    if cur_game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel <= game.inargs.min_fuel:
        print("beta seeker out of fuel", file=file)
    if cur_game_state[U.P1][U.SCORE] >= game.inargs.win_score[U.P1]:
        print("alpha reached Win Score", file=file)
    if cur_game_state[U.P2][U.SCORE]  >= game.inargs.win_score[U.P2]:
        print("beta reached Win Score", file=file)
    if cur_game_state[U.TURN_COUNT]  >= game.inargs.max_turns:
        print("max turns reached", file=file)

    print("\n====GAME FINISHED====\nWinner: {}\nScore: {}|{}\n=====================\n".format(winner, alpha_score, beta_score), file=file)


def log_game_to_file(game, logfile, actions=None):
    ''' add game state to game log file

    Args:
        logfile (str): path to game log file
    '''
    #Check that the file exists and create it if it doesn't exist
    if not os.path.isfile(logfile):
        with open(logfile, 'w') as f:
            #Write header with date and time
            f.write("Game Log File\n")
            f.write("Date: {}\n".format(datetime.datetime.now()))
            f.close()

    with open(logfile, 'a') as f:
        #Get the turn number and phase
        turn = game.game_state[U.TURN_COUNT]
        phase = game.game_state[U.TURN_PHASE]
        game_done = game.game_state[U.GAME_DONE]

        #If the game is done, then print the final score and winner
        if game_done:
            #Print final engagement outcomes
            print_engagement_outcomes(game.engagement_outcomes, file=f)
            print_game_info(game, file=f)
            print_endgame_status(game, file=f)

            #Close file and return
            f.close()
            return
        else:
            #Game is not done, so print latest info
            #If Turn Phase is Movement and Turn Count >1 print the engagement outcomes from the previous turn
            if phase == U.MOVEMENT:
                if turn >= 1:
                    print_engagement_outcomes(game.engagement_outcomes, file=f) #print engagement outcomes from previous turn
                    print_game_info(game, file=f) #print the token states at end of turn
                #Print the current turn number, phase, and scores
                print("\n<==== Turn: {} | Phase: {} ====>".format(turn, phase), file=f)
                print_scores(game, file=f)
                print_actions(actions, file=f) #print the selected movements before they are enacted
            elif phase == U.ENGAGEMENT:
                print("\n<==== Turn: {} | Phase: {} ====>".format(turn, phase), file=f)
                #If phase in engagement, then tokens have just completed the movement phase. Print the token states to see where they moved
                print_game_info(game, file=f) #print the token states before as the engagements are selected
                print_actions(actions, file=f) #print the selected engagements before they are enacted
            elif phase == U.DRIFT:
                print("\n<==== Turn: {} | Phase: {} ====>".format(turn, phase), file=f) #This should never really get called...
            else:
                print("Unrecognized game phase {}".format(phase), file=f)
                raise ValueError("Unrecognized game phase {}".format(phase))
            f.close()
            return