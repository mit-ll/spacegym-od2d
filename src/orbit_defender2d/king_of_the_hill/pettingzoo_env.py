# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# PettingZoo Environment Wrapper for King of the Hill

# Note: this is moved to its own source file so that
# pettingzoo_env is dependent upon koth, but koth is not dependent on pettingzoo_env.
# This allows for easier environment versioning by hard-coding version-
# dependent parameters here (e.g. board size, num tokens, init fuel, etc.)
# while allowing for full flexibility of the underlying koth game

from tabnanny import verbose
import numpy as np
import json
import pygame as pg

from pathlib import Path
#from gymnasium import spaces
from gym import spaces
from collections import namedtuple, OrderedDict
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
from pygame import gfxdraw      # needs its import to work as pg.gfxdraw for some reason

import orbit_defender2d.utils.utils as U
#import orbit_defender2d.king_of_the_hill.default_game_parameters as DGP
#import orbit_defender2d.king_of_the_hill.default_game_parameters_old as DGP
import orbit_defender2d.king_of_the_hill.default_game_parameters as DGP
import orbit_defender2d.king_of_the_hill.utils_for_json_display as UJD
import orbit_defender2d.king_of_the_hill.game_server as GS
import orbit_defender2d.king_of_the_hill.render_controls as RC
from orbit_defender2d.king_of_the_hill import koth

# observation space components
TokenComponentSpaces = namedtuple('TokenComponentSpaces', ['own_piece', 'role', 'position', 'fuel', 'ammo'])

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

# observation space flat encoding
# Note: hard-coding and then cross checking in order
# to avoid in-advertant observation space dimension changes
# due changes in dependent variable
N_BITS_OBS_SCORE = 12  # assumes max abs(score) of 1000 -> ownership bit + sign bit + 10 bits binary
N_BITS_OBS_TURN_COUNT = len(U.int2bitlist(int(DGP.MAX_TURNS)))  # assumes max turns of 100 -> 7 bits binary
N_BITS_OBS_TURN_PHASE = 3  # assumes 3 phases per turn (move, engage, drift)
N_BITS_OBS_HILL = int(DGP.NUM_SPACES+1+1) #number of spaces plus 1 plus a boolen for owner of hill
N_BITS_OBS_SCOREBOARD = N_BITS_OBS_TURN_PHASE + N_BITS_OBS_TURN_COUNT + N_BITS_OBS_SCORE + N_BITS_OBS_SCORE + N_BITS_OBS_HILL + N_BITS_OBS_HILL # concate turn_phase, turn_count, own_score, opponent_score, own_hill, opponent_hill
N_BITS_OBS_OWN_PIECE = 1  # 0=opponent, 1=own piece
N_BITS_OBS_ROLE = 2  # assumes 2 token roles: 0=seeker, 1=bludger, encoded as one-hot
N_BITS_OBS_POSITION = int(DGP.NUM_SPACES)+1 #number of spaces plus 1, TODO: Note sure why plus 1...
N_BITS_OBS_FUEL = 7  # assumes max fuel of 100 -> 7 bits binary
N_BITS_OBS_AMMO = len(U.int2bitlist(int(max(DGP.INIT_AMMO[U.P1][U.SEEKER],DGP.INIT_AMMO[U.P2][U.BLUDGER]))))  # assumes max ammo of 1 -> 1 bit binary
N_BITS_OBS_PER_TOKEN = N_BITS_OBS_OWN_PIECE + N_BITS_OBS_ROLE + N_BITS_OBS_POSITION + N_BITS_OBS_FUEL + N_BITS_OBS_AMMO  # number of bits for a single token observation own_piece + role + position + fuel + ammo
#N_BITS_OBS_TOKENS_PER_PLAYER = 814  # total number of bits for all of one player's tokens, num_tokens * N_BITS_OBS_PER_TOKEN
N_BITS_OBS_TOKENS_PER_PLAYER = N_BITS_OBS_PER_TOKEN * int(max(DGP.NUM_TOKENS_PER_PLAYER[U.P1],DGP.NUM_TOKENS_PER_PLAYER[U.P2])) #number of tokens per player * bits per token
N_BITS_OBS_PER_PLAYER = N_BITS_OBS_SCOREBOARD + 2 * N_BITS_OBS_TOKENS_PER_PLAYER # total number of bits for each player's complete observation, scoreboard + tokens*2

# cross-check hard-coded bit sizes with variables upon which they depend
assert N_BITS_OBS_ROLE == len(U.PIECE_ROLES)
assert N_BITS_OBS_POSITION == int(DGP.NUM_SPACES)+1  #Number of spaces on the board, not counting the center, then have to add 1, not sure why...
assert N_BITS_OBS_FUEL == max(
    len(U.int2bitlist(int(DGP.INIT_FUEL[U.P1][U.SEEKER]))),
    len(U.int2bitlist(int(DGP.INIT_FUEL[U.P1][U.BLUDGER]))),
    len(U.int2bitlist(int(DGP.INIT_FUEL[U.P2][U.SEEKER]))),
    len(U.int2bitlist(int(DGP.INIT_FUEL[U.P2][U.BLUDGER]))))
assert N_BITS_OBS_AMMO == max(
    len(U.int2bitlist(int(DGP.INIT_AMMO[U.P1][U.SEEKER]))),
    len(U.int2bitlist(int(DGP.INIT_AMMO[U.P1][U.BLUDGER]))),
    len(U.int2bitlist(int(DGP.INIT_AMMO[U.P2][U.SEEKER]))),
    len(U.int2bitlist(int(DGP.INIT_AMMO[U.P2][U.BLUDGER]))))
assert N_BITS_OBS_PER_TOKEN == N_BITS_OBS_OWN_PIECE + N_BITS_OBS_ROLE + N_BITS_OBS_POSITION + N_BITS_OBS_FUEL + N_BITS_OBS_AMMO

# action space flat encoding
# Note: hard-coding and then cross checking in order
# to avoid inadvertent observation space dimension changes
N_BITS_ACT_PER_TOKEN = len(U.MOVEMENT_TYPES) + len(U.ENGAGEMENT_TYPES)*int(max(DGP.NUM_TOKENS_PER_PLAYER[U.P1],DGP.NUM_TOKENS_PER_PLAYER[U.P2]))  #should be movement types + engagement types*tokens per player
N_BITS_ACT_PER_PLAYER = N_BITS_ACT_PER_TOKEN * int(max(DGP.NUM_TOKENS_PER_PLAYER[U.P1],DGP.NUM_TOKENS_PER_PLAYER[U.P2])) #should be tokens per player * bits per token action

def env(rllib_env_config=None):
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recomend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_env(rllib_env_config)
    env = wrappers.CaptureStdoutWrapper(env)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(rllib_env_config=None):
    '''
    To support the AEC API, the raw_env() function just uses the parallel_to_aec
    function to convert from a ParallelEnv to an AEC env

    See: https://www.pettingzoo.ml/environment_creation#example-custom-parallel-environment
    '''
    env = parallel_env(rllib_env_config)
    env = parallel_to_aec(env)
    return env


def cart2pol(x, y):
    '''Convert cartesian coordinates to polar coordinates (in degrees)'''
    r = np.sqrt(x ** 2 + y ** 2)
    a = np.degrees(np.arctan2(y, x))
    return r, a


def pol2cart(r, a, c):
    '''Convert polar coordinates (in degrees) to cartesian coordinates'''
    x = r * np.cos(np.radians(a)) + c[0]
    y = r * np.sin(np.radians(a)) + c[1]
    return x, y


class parallel_env(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "rps_v1"}

    def __init__(self, rllib_env_config=None):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.

        Observation space is composed of observation, that observes the board state, and 
        action_mask, that says which actions are legal. For comparison, see:
        https://github.com/PettingZoo-Team/PettingZoo/blob/master/pettingzoo/classic/chess/chess_env.py
        '''

        # instantiate gameboard
        self.kothgame = koth.KOTHGame(**GAME_PARAMS._asdict())

        # get agent names from game object
        self.possible_agents = self.kothgame.player_names
        # self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # get action and observation space information
        self.act_space_info = KOTHActionSpaces(game=self.kothgame)
        self.obs_space_info = KOTHObservationSpaces(game=self.kothgame)
        self.n_tokens_per_player = self.act_space_info.n_tokens_per_player

        # Action and Observation space formatted to PettingZoo API
        self.action_spaces = {
            agent: self.act_space_info.per_player for
            agent in self.possible_agents}
        self.observation_spaces = {agent: spaces.Dict({
            'observation': self.obs_space_info.flat_per_player,
            'action_mask': self.act_space_info.mask_per_player
        }) for agent in self.possible_agents}

        # Legal actions at current game state
        self.legal_actions = None

	    #Flag to track whether we are logging output
        self.render_json = None

        #Flag to track if game is over
        self.gameover = False

        if rllib_env_config is not None:
            self.workerid = rllib_env_config.worker_index
        else:
            self.workerid = None

        # Rendering variables
        # Rendering not always used
        self.render_active = False
        self._render_mode = None
        self._screen = None

        # fonts
        self._font = None
        self._font_bold = None
        self._large_font = None
        self._large_font_bold = None
        self._small_font_bold = None
        self._semi_large_font_bold = None
        self._very_large_font_bold = None

        # display dimensions
        self._x_dim = 1280  # 880
        self._y_dim = 720  # 560
        self._ring_count = self.kothgame.inargs.max_ring - self.kothgame.inargs.min_ring + 1
        self._margins = (10, 10)
        self._board_r = (self._y_dim - self._margins[1]) / 2
        self._board_c = (self._board_r + self._margins[0], self._y_dim / 2)
        self._earth_rotation = 0

        # display colors
        self._colors = {'white': (255, 255, 255), 'faint_green': (220, 255, 220), 'black': (0, 0, 0),
                        'gray': (150, 150, 150), 'dark_gray': (100, 100, 100), 'aqua': (0, 180, 180),
                        'red': (200, 0, 120), 'dull_aqua': (120, 150, 150), 'dull_red': (150, 120, 140),
                        'dark_aqua': (0, 80, 80), 'dark_red': (80, 0, 50), 'light_green': (180, 255, 180),
                        'light_yellow': (255, 255, 200), 'light_aqua': (180, 255, 255), 'light_red': (255, 150, 180)}
        self._bg_color = self._colors['black']
        self._board_color = self._colors['white']
        self._title_color = self._colors['white']
        self._text_color = self._colors['faint_green']
        self._p1_color = self._colors['aqua']
        self._p1_color_dark = self._colors['dark_aqua']
        self._p2_color = self._colors['red']
        self._p2_color_dark = self._colors['dark_red']
        self._null_color = self._colors['gray']
        self._null_color_dark = self._colors['dark_gray']

        # actions and outcomes for display
        self.actions = None
        self._eg_outcomes_phase = False

        # program flow and user control
        self._is_paused = True
        self._latency = 5000  # milliseconds between displaying turn phases
        self._min_latency = 500  # milliseconds between displaying turn phases
        self._buttons_active = False
        self._button_panel = None
        self._button_size = self._x_dim // 20

    def enable_render(self, mode):
        '''
        Initializes pygame, display for rendering, and fonts
        '''
        # initialize pygame library and display window
        pg.init()
        self.render_active = True
        self._render_mode = mode
        self._screen = pg.display.set_mode((self._x_dim, self._y_dim))
        pg.display.set_caption('Orbit Defender')

        # initialize fonts
        very_large_font_size = 48
        large_font_size = 28
        semi_large_font_size = 24
        font_size = 16
        small_font_size = 14
        self._font = pg.font.SysFont(pg.font.get_default_font(), font_size)
        self._font_bold = pg.font.SysFont(pg.font.get_default_font(), font_size, True)
        self._large_font = pg.font.SysFont(pg.font.get_default_font(), large_font_size)
        self._large_font_bold = pg.font.SysFont(pg.font.get_default_font(), large_font_size, True)
        self._small_font_bold = pg.font.SysFont(pg.font.get_default_font(), small_font_size, True)
        self._semi_large_font_bold = pg.font.SysFont(pg.font.get_default_font(), semi_large_font_size, True)
        self._very_large_font_bold = pg.font.SysFont(pg.font.get_default_font(), very_large_font_size, True)

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it opens
        up a graphical window to display the game board, pieces, and game details.
        '''
        
        if mode == "json":
            print(f'inrender, worker is: {self.workerid}, flag is: {self.render_json}')
            if self.render_json is None:
                #Create file to write output to
                if self.workerid is not None:
                    filename = f'./render_output_{self.workerid}.json'
                else:
                    filename = f'./render_single_output.json'
                
                self.render_json = open(filename, 'w')

                #Get game state & format it for sending
                gs_formatted = UJD.format_response_message(GS.GAME_RESET_RESP, GS.GAME_RESET, self.kothgame)
                json.dump(gs_formatted, self.render_json)
                self.render_json.write('\n')
                print(f"Wrote {GS.GAME_RESET_RESP} message")	


        if (mode == "human" or mode == "debug") and not self.render_active:
            # initializes display and fonts, and sets self.render_active to True
            self.enable_render(mode)

        if mode == "human" or mode == "debug":
            self._screen.fill(self._bg_color)
            self._draw_earth()
            self._draw_board()
            self._draw_details()
            self._draw_tokens()
            pg.display.update()

        if mode == "human":
            pg.time.wait(self._latency)
        elif mode == "debug":
            self._handle_events()

    def _draw_board(self):
        '''Draws the game board with numbered sectors, highlighting the goal sector for each player'''
        # tracks which sector is being drawn
        sector_count = 1

        g1_st_angle = 0
        g1_end_angle = 0
        g1_r_min = 0
        g1_r_max = 0

        g2_st_angle = 0
        g2_end_angle = 0
        g2_r_min = 0
        g2_r_max = 0

        g_arc_rect_inner = None
        g_arc_rect_outer = None

        g_line_width = 3

        # draw board ring by ring from the center outwards
        for ring in range(1, self._ring_count + 1):
            ring_r_min = int(ring / (self._ring_count + 1) * self._board_r)
            ring_r_max = int((ring + 1) / (self._ring_count + 1) * self._board_r)
            arc_rect_inner = ((self._board_c[0] - ring_r_min),
                              (self._board_c[1] - ring_r_min),
                              2 * ring_r_min + 3,
                              2 * ring_r_min + 3)
            arc_rect_outer = ((self._board_c[0] - ring_r_max),
                              (self._board_c[1] - ring_r_max),
                              2 * ring_r_max + 3,
                              2 * ring_r_max + 3)
            ring_sectors = 2 ** ring
            num_r = ring_r_max - 10

            # draw rings
            pg.gfxdraw.aacircle(self._screen, int(self._board_c[0]),
                                int(self._board_c[1]), ring_r_min, self._board_color)
            pg.gfxdraw.aacircle(self._screen, int(self._board_c[0]),
                                int(self._board_c[1]), ring_r_max, self._board_color)

            # draw appropriate number of sectors for current ring
            for sector_idx in range(ring_sectors):
                st_angle = 2 * np.pi * (1 - (sector_idx / ring_sectors))
                end_angle = 2 * np.pi * (1 - ((sector_idx + 1) / ring_sectors))
                num_angle = end_angle - ((end_angle - st_angle) / 7)

                # mark each player's goal sector, don't
                if sector_count == self.kothgame.game_state[U.GOAL1]:
                    g1_st_angle = st_angle
                    g1_end_angle = end_angle
                    g1_r_min = ring_r_min
                    g1_r_max = ring_r_max
                    g_arc_rect_inner = arc_rect_inner
                    g_arc_rect_outer = arc_rect_outer
                elif sector_count == self.kothgame.game_state[U.GOAL2]:
                    g2_st_angle = st_angle
                    g2_end_angle = end_angle
                    g2_r_min = ring_r_min
                    g2_r_max = ring_r_max
                    g_arc_rect_inner = arc_rect_inner
                    g_arc_rect_outer = arc_rect_outer

                # draw inner/outer arcs of each sector
                # uses polar coordinates
                # pg.draw.arc(self._screen, self._board_color, arc_rect_inner, st_angle, end_angle, width=line_width)
                # pg.draw.arc(self._screen, self._board_color, arc_rect_outer, st_angle, end_angle, width=line_width)

                # draw lines on either side of each sector
                # uses cartesian coordinates (must convert)
                pg.draw.aaline(self._screen, self._board_color,
                               pol2cart(ring_r_min, np.degrees(st_angle), self._board_c),
                               pol2cart(ring_r_max, np.degrees(st_angle), self._board_c))
                pg.draw.aaline(self._screen, self._board_color,
                               pol2cart(ring_r_min, np.degrees(end_angle), self._board_c),
                               pol2cart(ring_r_max, np.degrees(end_angle), self._board_c))

                # number each sector
                text = self._font.render(str(sector_count), True, self._board_color)
                self._screen.blit(text,
                                  pol2cart(num_r, np.degrees(num_angle), (self._board_c[0] - 5, self._board_c[1] - 5)))

                sector_count += 1

        # highlight alpha goal sector
        # draw inner/outer arcs of each sector
        # uses polar coordinates
        pg.draw.arc(self._screen, self._p1_color, g_arc_rect_inner, -g1_st_angle, -g1_end_angle, width=g_line_width)
        pg.draw.arc(self._screen, self._p1_color, g_arc_rect_outer, -g1_st_angle, -g1_end_angle, width=g_line_width)

        # draw lines on either side of each sector
        # uses cartesian coordinates (must convert)
        pg.draw.line(self._screen, self._p1_color, pol2cart(g1_r_min, np.degrees(g1_st_angle), self._board_c),
                     pol2cart(g1_r_max, np.degrees(g1_st_angle), self._board_c), width=g_line_width)
        pg.draw.line(self._screen, self._p1_color, pol2cart(g1_r_min, np.degrees(g1_end_angle), self._board_c),
                     pol2cart(g1_r_max, np.degrees(g1_end_angle), self._board_c), width=g_line_width)

        # highlight beta goal sector
        # draw inner/outer arcs of each sector
        # uses polar coordinates
        pg.draw.arc(self._screen, self._p2_color, g_arc_rect_inner, -g2_st_angle, -g2_end_angle, width=g_line_width)
        pg.draw.arc(self._screen, self._p2_color, g_arc_rect_outer, -g2_st_angle, -g2_end_angle, width=g_line_width)

        # draw lines on either side of each sector
        # uses cartesian coordinates (must convert)
        pg.draw.line(self._screen, self._p2_color, pol2cart(g2_r_min, np.degrees(g2_st_angle), self._board_c),
                     pol2cart(g2_r_max, np.degrees(g2_st_angle), self._board_c), width=g_line_width)
        pg.draw.line(self._screen, self._p2_color, pol2cart(g2_r_min, np.degrees(g2_end_angle), self._board_c),
                     pol2cart(g2_r_max, np.degrees(g2_end_angle), self._board_c), width=g_line_width)

    def _draw_details(self):
        '''
        Draws game details to the right of the game board; details include:
         - Score
         - Turn Count
         - Turn Phase
         - Player Titles
         - Details of Each Token:
            - Name / Type
            - Movement
            - Engagement
                - Kinetic shot marked by filled triangles with a line
                - Collision marked by outlined triangles with a line
                - Guard marked by square and shield (square displays piece being guarded, shield displayed on target
                  with # of pieces guarding it)
                - Engagements that can't take effect are grayed out
                - Engagements that can take effect, but fail, are grayed out and marked with an x
            - Fuel
            - Ammo
        '''
        # find horizontal center for displaying details and necessary font sizes
        x_mid = self._x_dim * 3 / 4
        b_font_size = self._font_bold.size(' ')
        l_font_size = self._large_font.size(' ')
        lb_font_size = self._large_font_bold.size(' ')

        # display score
        score_title = self._large_font_bold.render("Score:", True, self._title_color)
        self._screen.blit(score_title, (x_mid - (3 * lb_font_size[0]), self._margins[1]))

        divider = self._large_font_bold.render("|", True, self._title_color)
        self._screen.blit(divider, (x_mid + (3 * lb_font_size[0]), self._margins[1] + lb_font_size[1]))

        p1_score = self._large_font.render(str(self.kothgame.game_state[U.P1][U.SCORE]), True, self._p1_color)
        l_score_len = len(str(self.kothgame.game_state[U.P1][U.SCORE]))
        self._screen.blit(p1_score, (x_mid - ((l_score_len + 2) * l_font_size[0]), self._margins[1] + lb_font_size[1]))

        p2_score = self._large_font.render(str(self.kothgame.game_state[U.P2][U.SCORE]), True, self._p2_color)
        self._screen.blit(p2_score, (x_mid + (6 * l_font_size[0]), self._margins[1] + lb_font_size[1]))

        # display turn count
        turn_str = str(self.kothgame.game_state[U.TURN_COUNT])
        turns_title = self._large_font_bold.render("Turn:", True, self._title_color)
        self._screen.blit(turns_title, (x_mid - ((len(turn_str) + 3) * l_font_size[0]),
                                        self._margins[1] + (2.5 * lb_font_size[1])))
        turn = self._large_font.render(turn_str, True, self._text_color)
        self._screen.blit(turn, (x_mid + ((9 - len(turn_str)) * l_font_size[0]),
                                 self._margins[1] + (2.5 * lb_font_size[1])))

        # display turn phase
        phase_str = str(self.kothgame.game_state[U.TURN_PHASE]).capitalize()
        phase_title = self._large_font_bold.render("Phase: ", True, self._title_color)

        # marks engagement outcomes phase of display
        if phase_str == "Drift" and self.actions:
            phase_str = "Engagement"
            phase = self._large_font.render("Outcomes", True, self._text_color)
            self._screen.blit(phase, (x_mid - (7 * l_font_size[0]), self._margins[1] + (5 * lb_font_size[1])))

        self._screen.blit(phase_title, (x_mid - (2 * (len(phase_str) - 1) * l_font_size[0]),
                                        self._margins[1] + (4 * lb_font_size[1])))
        phase = self._large_font.render(phase_str, True, self._text_color)
        self._screen.blit(phase, (x_mid + (2 * (8 - len(phase_str)) * l_font_size[0]),
                                  self._margins[1] + (4 * lb_font_size[1])))

        # display legend
        legend_pos = (x_mid + (18 * lb_font_size[0]), self._margins[1])
        legend_rect = (legend_pos[0],  # left
                       legend_pos[1],  # top
                       self._x_dim - self._margins[0] - legend_pos[0],  # width
                       5 * lb_font_size[1])  # height
        pg.draw.rect(self._screen, self._title_color, legend_rect, width=2)
        legend_title = self._large_font_bold.render("Key", True, self._title_color)
        self._screen.blit(legend_title, (legend_pos[0] + (legend_rect[2] / 2) - (4 * lb_font_size[0]),
                                         self._margins[1] + (0.2 * lb_font_size[1])))

        filled_poly1 = ((legend_pos[0] + lb_font_size[0], legend_pos[1] + (1.7 * lb_font_size[1])),
                        ((legend_pos[0] + (2.8 * lb_font_size[0])), legend_pos[1] + (1.45 * lb_font_size[1])),
                        (legend_pos[0] + (2.8 * lb_font_size[0]), legend_pos[1] + (1.95 * lb_font_size[1])))
        filled_poly2 = ((legend_pos[0] + (5.6 * lb_font_size[0]), legend_pos[1] + (1.7 * lb_font_size[1])),
                        (legend_pos[0] + (3.8 * lb_font_size[0]), legend_pos[1] + (1.45 * lb_font_size[1])),
                        (legend_pos[0] + (3.8 * lb_font_size[0]), legend_pos[1] + (1.95 * lb_font_size[1])))
        description = self._semi_large_font_bold.render("= Kinetic Shot", True, self._text_color)
        pg.gfxdraw.filled_polygon(self._screen, filled_poly1, self._title_color)
        pg.gfxdraw.filled_polygon(self._screen, filled_poly2, self._title_color)
        self._screen.blit(description,
                          (legend_pos[0] + (7.4 * lb_font_size[0]), legend_pos[1] + (1.35 * lb_font_size[1])))

        poly1 = ((legend_pos[0] + lb_font_size[0], legend_pos[1] + (2.5 * lb_font_size[1])),
                 ((legend_pos[0] + (2.8 * lb_font_size[0])), legend_pos[1] + (2.25 * lb_font_size[1])),
                 (legend_pos[0] + (2.8 * lb_font_size[0]), legend_pos[1] + (2.75 * lb_font_size[1])))
        poly2 = ((legend_pos[0] + (5.6 * lb_font_size[0]), legend_pos[1] + (2.5 * lb_font_size[1])),
                 (legend_pos[0] + (3.8 * lb_font_size[0]), legend_pos[1] + (2.25 * lb_font_size[1])),
                 (legend_pos[0] + (3.8 * lb_font_size[0]), legend_pos[1] + (2.75 * lb_font_size[1])))
        description = self._semi_large_font_bold.render("= Collision", True, self._text_color)
        pg.gfxdraw.aapolygon(self._screen, poly1, self._title_color)
        pg.gfxdraw.aapolygon(self._screen, poly2, self._title_color)
        self._screen.blit(description,
                          (legend_pos[0] + (7.4 * lb_font_size[0]), legend_pos[1] + (2.15 * lb_font_size[1])))

        guard_rect = (legend_pos[0] + (1.8 * lb_font_size[0]),  # left
                      legend_pos[1] + (3 * lb_font_size[1]),  # top
                      3 * lb_font_size[0],  # width
                      1.2 * b_font_size[1])  # height
        num_pos = (guard_rect[0] + (0.7 * lb_font_size[0]),
                   guard_rect[1] + (0 * lb_font_size[1]))
        num = self._semi_large_font_bold.render("#", True, self._text_color)
        description = self._semi_large_font_bold.render("= Guard Target", True, self._text_color)
        pg.draw.rect(self._screen, self._title_color, guard_rect, width=1)
        self._screen.blit(num, num_pos)
        self._screen.blit(description,
                          (legend_pos[0] + (7.4 * lb_font_size[0]), legend_pos[1] + (2.95 * lb_font_size[1])))

        shield_center = (legend_pos[0] + (3.3 * lb_font_size[0]), legend_pos[1] + (4.2 * lb_font_size[1]))
        shield_poly = ((shield_center[0] - (0.15 * lb_font_size[0]),  # top left divot
                       shield_center[1] - (0.6 * b_font_size[1])),
                       (shield_center[0], shield_center[1] - (0.7 * b_font_size[1])),  # top
                       (shield_center[0] + (0.15 * lb_font_size[0]),  # top right divot
                       shield_center[1] - (0.6 * b_font_size[1])),
                       (shield_center[0] + (1.4 * lb_font_size[0]),  # top right
                       shield_center[1] - (0.4 * b_font_size[1])),
                       (shield_center[0] + (1.1 * lb_font_size[0]),  # bottom right
                       shield_center[1] + (0.3 * b_font_size[1])),
                       (shield_center[0], shield_center[1] + (0.7 * b_font_size[1])),  # bottom
                       (shield_center[0] - (1.1 * lb_font_size[0]),  # bottom left
                       shield_center[1] + (0.3 * b_font_size[1])),
                       (shield_center[0] - (1.4 * lb_font_size[0]),  # top left
                       shield_center[1] - (0.4 * b_font_size[1])))
        num_pos = (shield_center[0] - (0.7 * lb_font_size[0]),
                   shield_center[1] - (0.25 * lb_font_size[1]))
        num = self._semi_large_font_bold.render("#", True, self._text_color)
        description = self._semi_large_font_bold.render("= Guard Count", True, self._text_color)
        pg.gfxdraw.aapolygon(self._screen, shield_poly, self._title_color)
        self._screen.blit(num, num_pos)
        self._screen.blit(description,
                          (legend_pos[0] + (7.4 * lb_font_size[0]), legend_pos[1] + (3.75 * lb_font_size[1])))

        # display team / player titles
        p1_title = self._large_font_bold.render("Alpha", True, self._p1_color)
        self._screen.blit(p1_title, (x_mid - (24 * lb_font_size[0]), self._margins[1] + (5.5 * lb_font_size[1])))

        p2_title = self._large_font_bold.render("Beta", True, self._p2_color)
        self._screen.blit(p2_title, (x_mid + (20 * lb_font_size[0]), self._margins[1] + (5.5 * lb_font_size[1])))

        guarded_tokens = dict()  # tracks which tokens are being guarded, stores data for displaying guard counts
        attacked_tokens = []  # tracks which tokens are under attack
        inactive_tokens = []  # tracks which tokens are inactive
        # track each token that is under attack and / or inactive

        #if hasattr(self, 'verbose_actions'):
        #    self.actions = self.verbose_actions #See if this works... may break other ways to use this env though...

        if self.kothgame.game_state[U.TURN_PHASE] == "engagement" and not self._eg_outcomes_phase:
            for token_name, token_state in self.kothgame.token_catalog.items():
                #There should be actions that are in koth touple format, but they are probably in gym format instead
                if self.actions:
                    #print("\n<==== Turn: {} | Phase: {} ====>".format(
                    #    self.kothgame.game_state[U.TURN_COUNT], 
                    #    self.kothgame.game_state[U.TURN_PHASE]))
                    #print("Token: {} | Action: {}".format(token_name, self.actions[token_name].action_type))
                    if self.actions[token_name].action_type == "shoot" or self.actions[token_name].action_type == "collide":
                        attacked_tokens.append(token_name)
                else: #TODO: I think this else can be deleted now...
                    if hasattr(self, 'verbose_actions'):
                        #print("Trying engagement with verbose actions")
                        #print("Token name: {}   token action: {}".format(token_name, self.verbose_actions[token_name].action_type))
                        if self.verbose_actions[token_name].action_type == "shoot" or self.verbose_actions[token_name].action_type == "collide":
                            attacked_tokens.append(token_name)
                if token_state.satellite.fuel == self.kothgame.inargs.min_fuel:
                    inactive_tokens.append(token_name)

        # display details of each token
        for token_name, token_state in self.kothgame.token_catalog.items():
            if token_state.position == 0: #If token is in the center, don't display it
                continue
            # determine token player (influences color and horizontal alignment)
            split_name = token_name.split(':')
            if split_name[0] == "alpha":
                color = self._p1_color
                engagement_color = self._p1_color
                player_x_mid = x_mid - (23 * lb_font_size[0])
            else:
                color = self._p2_color
                engagement_color = self._p2_color
                player_x_mid = x_mid + (20 * lb_font_size[0])

            # if no fuel remaining (token is inactive), draw without player color
            if token_state.satellite.fuel == self.kothgame.inargs.min_fuel:
                color = self._null_color
                subtitle_color = self._null_color
                cur_inactive = True
            else:
                subtitle_color = self._title_color
                cur_inactive = False

            # determine token type and number (influences vertical alignment)
            if split_name[1] == "seeker":
                player_y_mid = self._margins[1] + (8.5 * lb_font_size[1])
                name_short = "Seeker 0"
            else:
                t_num = int(token_name[-1])
                if t_num == 0:
                    t_num = 10
                name_short = "Bludger " + str(t_num)
                player_y_mid = self._margins[1] + (8.5 * lb_font_size[1]) + (t_num * (4 * b_font_size[1]))

            # display bounding box
            pg.draw.rect(self._screen, color, pg.Rect(player_x_mid - (10 * b_font_size[0]),  # left
                                                      player_y_mid - (3 * b_font_size[1]),  # top
                                                      36 * b_font_size[0],  # width
                                                      4 * b_font_size[1]), width=2)  # height

            # display token name
            t_name = self._font_bold.render(name_short, True, color)
            self._screen.blit(t_name, (player_x_mid + (14 - (len(name_short)) * b_font_size[0] / 2),
                                       player_y_mid - (2.5 * b_font_size[1])))

            # display movement
            t_move_title = self._font_bold.render("Position:", True, subtitle_color)
            self._screen.blit(t_move_title,
                              (player_x_mid - (9 * b_font_size[0]), player_y_mid - (1.4 * b_font_size[1])))

            pos = token_state.position
            move_str = ""
            if self.actions and self.kothgame.game_state[U.TURN_PHASE] == "movement":
                    # determine movement type
                    if self.actions[token_name].action_type == "prograde":
                        move_str = " > " + str(self.kothgame.board_grid.get_prograde_sector(pos))
                    elif self.actions[token_name].action_type == "retrograde":
                        move_str = " > " + str(self.kothgame.board_grid.get_retrograde_sector(pos))
                    elif self.actions[token_name].action_type == "radial_in":
                        move_str = " > " + str(self.kothgame.board_grid.get_radial_in_sector(pos))
                    elif self.actions[token_name].action_type == "radial_out":
                        move_str = " > " + str(self.kothgame.board_grid.get_radial_out_sector(pos))
                    else:
                        pass
            # display engagement or engagement outcomes
            elif self.actions and hasattr(self.actions[token_name], "target") and (self.kothgame.game_state[U.TURN_PHASE] == "engagement" or self._eg_outcomes_phase):
                # determine target
                target_name = self.actions[token_name].target
                target = target_name.split(':')

                # when on engagement outcome display phase, determine whether action was successful
                inactive_target = target_name in inactive_tokens
                eg_failed = self._eg_outcomes_phase and (type(self.actions[token_name]) == U.EngagementOutcomeTuple
                                                         and not self.actions[token_name].success)

                # determine engagement type
                if self.actions[token_name].action_type == "shoot":
                    if eg_failed or cur_inactive:
                        engagement_color = self._null_color
                    # determine attacker
                    # shoot indicator starts at top corner of attacker details, ends at bottom corner of target details
                    # marked by filled triangles
                    if split_name[0] == "alpha":
                        line_start = (x_mid - (5.5 * lb_font_size[0]), player_y_mid - (1.2 * lb_font_size[1]))
                        line_end = (x_mid + (12 * lb_font_size[0]),
                                    self._margins[1] + (8.7 * lb_font_size[1]) +
                                    (int(target[2]) * (4 * b_font_size[1])))
                        poly1 = (line_start,
                                 (line_start[0] - (1.8 * lb_font_size[0]), line_start[1] + (0.5 * b_font_size[1])),
                                 (line_start[0] - (1.8 * lb_font_size[0]), line_start[1] - (0.5 * b_font_size[1])))
                        poly2 = ((line_end[0] + (1.8 * lb_font_size[0]), line_end[1]),
                                 (line_end[0], line_end[1] + (0.5 * b_font_size[1])),
                                 (line_end[0], line_end[1] - (0.5 * b_font_size[1])))
                    else:
                        line_start = (x_mid + (12 * lb_font_size[0]), player_y_mid - (1.2 * lb_font_size[1]))
                        line_end = (x_mid - (5.5 * lb_font_size[0]),
                                    self._margins[1] + (8.7 * lb_font_size[1]) +
                                    (int(target[2]) * (4 * b_font_size[1])))
                        poly1 = (line_start,
                                 (line_start[0] + (1.8 * lb_font_size[0]), line_start[1] + (0.5 * b_font_size[1])),
                                 (line_start[0] + (1.8 * lb_font_size[0]), line_start[1] - (0.5 * b_font_size[1])))
                        poly2 = ((line_end[0] - (1.8 * lb_font_size[0]), line_end[1]),
                                 (line_end[0], line_end[1] + (0.5 * b_font_size[1])),
                                 (line_end[0], line_end[1] - (0.5 * b_font_size[1])))

                    pg.draw.polygon(self._screen, engagement_color, poly1, width=0)
                    pg.draw.polygon(self._screen, engagement_color, poly2, width=0)
                    pg.draw.aaline(self._screen, engagement_color, line_start, line_end)

                    if eg_failed and not (cur_inactive or inactive_target):
                        cross_center = (line_start[0], line_start[1])
                        cross_size = lb_font_size[1] / 2
                        cross_start1 = pol2cart(cross_size / 2, 45, cross_center)
                        cross_start2 = pol2cart(cross_size / 2, 135, cross_center)
                        cross_end1 = pol2cart(cross_size / 2, 225, cross_center)
                        cross_end2 = pol2cart(cross_size / 2, 315, cross_center)
                        pg.draw.line(self._screen, engagement_color, cross_start1, cross_end1, width=2)
                        pg.draw.line(self._screen, engagement_color, cross_start2, cross_end2, width=2)
                        cross_center = (line_end[0], line_end[1])
                        cross_start1 = pol2cart(cross_size / 2, 45, cross_center)
                        cross_start2 = pol2cart(cross_size / 2, 135, cross_center)
                        cross_end1 = pol2cart(cross_size / 2, 225, cross_center)
                        cross_end2 = pol2cart(cross_size / 2, 315, cross_center)
                        pg.draw.aaline(self._screen, engagement_color, cross_start1, cross_end1)
                        pg.draw.aaline(self._screen, engagement_color, cross_start2, cross_end2)
                    elif self._eg_outcomes_phase:
                        inactive_tokens.append(target_name)

                elif self.actions[token_name].action_type == "collide":
                    if eg_failed or inactive_target or cur_inactive:
                        engagement_color = self._null_color
                    # determine attacker
                    # collide indicator starts at top corner of attacker details,
                    # ends at bottom corner of target details
                    # marked by outlined triangles
                    if split_name[0] == "alpha":
                        line_start = (x_mid - (5.5 * lb_font_size[0]), player_y_mid - (1.2 * lb_font_size[1]))
                        line_end = (x_mid + (12 * lb_font_size[0]),
                                    self._margins[1] + (8.7 * lb_font_size[1]) +
                                    (int(target[2]) * (4 * b_font_size[1])))
                        poly1 = (line_start,
                                 (line_start[0] - (1.8 * lb_font_size[0]), line_start[1] + (0.5 * b_font_size[1])),
                                 (line_start[0] - (1.8 * lb_font_size[0]), line_start[1] - (0.5 * b_font_size[1])))
                        poly2 = ((line_end[0] + (1.8 * lb_font_size[0]), line_end[1]),
                                 (line_end[0], line_end[1] + (0.5 * b_font_size[1])),
                                 (line_end[0], line_end[1] - (0.5 * b_font_size[1])))
                    else:
                        line_start = (x_mid + (12 * lb_font_size[0]), player_y_mid - (1.2 * lb_font_size[1]))
                        line_end = (x_mid - (5.5 * lb_font_size[0]),
                                    self._margins[1] + (8.7 * lb_font_size[1]) +
                                    (int(target[2]) * (4 * b_font_size[1])))
                        poly1 = (line_start,
                                 (line_start[0] + (1.8 * lb_font_size[0]), line_start[1] + (0.5 * b_font_size[1])),
                                 (line_start[0] + (1.8 * lb_font_size[0]), line_start[1] - (0.5 * b_font_size[1])))
                        poly2 = ((line_end[0] - (1.8 * lb_font_size[0]), line_end[1]),
                                 (line_end[0], line_end[1] + (0.5 * b_font_size[1])),
                                 (line_end[0], line_end[1] - (0.5 * b_font_size[1])))

                    pg.gfxdraw.aapolygon(self._screen, poly1, engagement_color)
                    pg.gfxdraw.aapolygon(self._screen, poly2, engagement_color)
                    pg.draw.aaline(self._screen, engagement_color, line_start, line_end)

                    if eg_failed and not (token_name in inactive_tokens or inactive_target):
                        cross_center = (line_start[0], line_start[1])
                        cross_size = lb_font_size[1] / 2
                        cross_start1 = pol2cart(cross_size / 2, 45, cross_center)
                        cross_start2 = pol2cart(cross_size / 2, 135, cross_center)
                        cross_end1 = pol2cart(cross_size / 2, 225, cross_center)
                        cross_end2 = pol2cart(cross_size / 2, 315, cross_center)
                        pg.draw.aaline(self._screen, engagement_color, cross_start1, cross_end1)
                        pg.draw.aaline(self._screen, engagement_color, cross_start2, cross_end2)
                        cross_center = (line_end[0], line_end[1])
                        cross_start1 = pol2cart(cross_size / 2, 45, cross_center)
                        cross_start2 = pol2cart(cross_size / 2, 135, cross_center)
                        cross_end1 = pol2cart(cross_size / 2, 225, cross_center)
                        cross_end2 = pol2cart(cross_size / 2, 315, cross_center)
                        pg.draw.aaline(self._screen, engagement_color, cross_start1, cross_end1)
                        pg.draw.aaline(self._screen, engagement_color, cross_start2, cross_end2)
                    elif self._eg_outcomes_phase:
                        inactive_tokens.append(token_name)
                        inactive_tokens.append(target_name)
                elif self.actions[token_name].action_type == "guard":
                    if eg_failed or inactive_target or cur_inactive or \
                            (self._eg_outcomes_phase and target_name not in attacked_tokens):
                        engagement_color = self._null_color
                    # determine defender
                    # guard indicator consists of an outlined square and shield
                    # outlined square contains # of target
                    # outlined shield on target contains # of tokens guarding it
                    if split_name[0] == "alpha":
                        shield_center = (x_mid - (10.2 * lb_font_size[0]),
                                         self._margins[1] + (int(target[2]) * (4 * b_font_size[1])) +
                                         (7.6 * lb_font_size[1]))
                        guard_rect = (x_mid - (7.4 * lb_font_size[0]),  # left
                                      player_y_mid - (1.2 * lb_font_size[1]),  # top
                                      3 * lb_font_size[0],  # width
                                      1.2 * b_font_size[1])  # height
                        num_pos = (guard_rect[0] + (1.5 * lb_font_size[0]) - ((len(target[2]) + 1) * b_font_size[0]),
                                   guard_rect[1] + (0.1 * lb_font_size[1]))
                    else:
                        shield_center = (x_mid + (32.9 * lb_font_size[0]), self._margins[1] +
                                         (int(target[2]) * (4 * b_font_size[1])) + (7.6 * lb_font_size[1]))
                        guard_rect = (x_mid + (11.3 * lb_font_size[0]),  # left
                                      player_y_mid - (1.2 * lb_font_size[1]),  # top
                                      3 * lb_font_size[0],  # width
                                      1.2 * b_font_size[1])  # height
                        num_pos = (guard_rect[0] + (1.5 * lb_font_size[0]) - ((len(target[2]) + 1) * b_font_size[0]),
                                   guard_rect[1] + (0.1 * lb_font_size[1]))

                    # draw outlined square and target #
                    pg.draw.rect(self._screen, engagement_color, guard_rect, width=1)

                    if target[2] == '0':
                        target_num = self._font_bold.render(' S', True, engagement_color)

                    else:
                        target_num = self._small_font_bold.render('B' + target[2], True, engagement_color)

                    self._screen.blit(target_num, num_pos)

                    # shield not displayed / guard count not incremented when guard cannot take effect
                    # to avoid confusion
                    if not inactive_target and not cur_inactive:
                        # do not increment guard count or draw shield if the engagement failed
                        if eg_failed:
                            # only draw cross on failed guard actions if the target is being attacked
                            if target_name in attacked_tokens:
                                cross_start1 = (guard_rect[0], guard_rect[1])
                                cross_end1 = (guard_rect[0] + (3 * lb_font_size[0]),
                                              guard_rect[1] + (1.2 * b_font_size[1]))
                                cross_start2 = (guard_rect[0], guard_rect[1] + (1.2 * b_font_size[1]))
                                cross_end2 = (guard_rect[0] + (3 * lb_font_size[0]), guard_rect[1])
                                pg.draw.aaline(self._screen, engagement_color, cross_start1, cross_end1)
                                pg.draw.aaline(self._screen, engagement_color, cross_start2, cross_end2)
                        else:
                            # increment guard_count of target token, include position and color data to be drawn later
                            if target_name in guarded_tokens:
                                guarded_tokens[target_name]['Num'] += 1
                                guarded_tokens[target_name]['NumPos'] = \
                                    (shield_center[0] - (0.4 * len(str(split_name[2])) * lb_font_size[0]),
                                     shield_center[1] - (0.45 * b_font_size[1]))
                            else:
                                guarded_tokens[target_name] = dict()
                                guarded_tokens[target_name]['Color'] = engagement_color
                                guarded_tokens[target_name]['Num'] = 1
                                guarded_tokens[target_name]['NumPos'] = \
                                    (shield_center[0] - (0.4 * len(str(split_name[2])) * lb_font_size[0]),
                                     shield_center[1] - (0.45 * b_font_size[1]))
                                guarded_tokens[target_name]['ShieldPoly'] = \
                                    ((shield_center[0] - (0.15 * lb_font_size[0]),  # top left divot
                                      shield_center[1] - (0.6 * b_font_size[1])),
                                     (shield_center[0], shield_center[1] - (0.7 * b_font_size[1])),  # top
                                     (shield_center[0] + (0.15 * lb_font_size[0]),  # top right divot
                                      shield_center[1] - (0.6 * b_font_size[1])),
                                     (shield_center[0] + (1.4 * lb_font_size[0]),  # top right
                                      shield_center[1] - (0.4 * b_font_size[1])),
                                     (shield_center[0] + (1.1 * lb_font_size[0]),  # bottom right
                                      shield_center[1] + (0.3 * b_font_size[1])),
                                     (shield_center[0], shield_center[1] + (0.7 * b_font_size[1])),  # bottom
                                     (shield_center[0] - (1.1 * lb_font_size[0]),  # bottom left
                                      shield_center[1] + (0.3 * b_font_size[1])),
                                     (shield_center[0] - (1.4 * lb_font_size[0]),  # top left
                                      shield_center[1] - (0.4 * b_font_size[1])))
                else:
                    pass
            elif self.kothgame.game_state[U.TURN_PHASE] == "drift" and not self.actions:
                # all tokens drift in prograde direction
                move_str = " > " + str(self.kothgame.board_grid.get_prograde_sector(pos))

            t_move = self._font.render(str(pos) + move_str, True, self._text_color)
            self._screen.blit(t_move, (player_x_mid + (8 * b_font_size[0]), player_y_mid - (1.4 * b_font_size[1])))

            # display fuel
            t_fuel_title = self._font_bold.render("Fuel:", True, subtitle_color)
            self._screen.blit(t_fuel_title,
                              (player_x_mid - (9 * b_font_size[0]), player_y_mid - (0.2 * b_font_size[1])))
            t_fuel = self._font.render(str(token_state.satellite.fuel), True, self._text_color)
            self._screen.blit(t_fuel, (player_x_mid, player_y_mid - (0.2 * b_font_size[1])))

            # display ammo if token is not seeker
            if "bludger" in token_name:
                t_ammo_title = self._font_bold.render("Ammo:", True, subtitle_color)
                self._screen.blit(t_ammo_title,
                                  (player_x_mid + (10 * b_font_size[0]), player_y_mid - (0.2 * b_font_size[1])))
                t_ammo = self._font.render(str(token_state.satellite.ammo), True, self._text_color)
                self._screen.blit(t_ammo,
                                  (player_x_mid + (23.5 * b_font_size[0]), player_y_mid - (0.2 * b_font_size[1])))

        # draw shields and guard counts for each guarded token
        for token in guarded_tokens.values():
            pg.gfxdraw.aapolygon(self._screen, token['ShieldPoly'], token['Color'])
            guard_count = self._font_bold.render(str(token['Num']), True, token['Color'])
            self._screen.blit(guard_count, token['NumPos'])

        # advance to engagement outcomes display phase when on engagement phase
        if self.kothgame.game_state[U.TURN_PHASE] == "engagement" and not self._eg_outcomes_phase:
            self._eg_outcomes_phase = True
        elif self._eg_outcomes_phase:
            self._eg_outcomes_phase = False
            self.actions = None

    def _draw_tokens(self):
        '''
        Draw each token in its appropriate sector:
         - Seekers marked by squares (level 3 within sector)
         - Bludgers marked by circles (Active: level 2 within sector, Inactive: level 1 within sector)
         - Inactive pieces (without fuel) are grayed out in the center
        '''
        sector_occupancies = dict()
        #b_font_size = self._font_bold.size(' ')
        b_font_size = self._large_font.size(' ')
        # store counts of necessary token types to track occupancy of each sector
        for token_name, token_state in self.kothgame.token_catalog.items():
            if token_state.position == 0: #If token is in the center, don't display it. This is used for asymmetric game.
                continue

            split_name = token_name.split(':')

            # initialize dictionaries for new sectors
            if token_state.position not in sector_occupancies:
                # counts of each type of token
                # 'AS': Alpha Seeker, 'ABA': Alpha Bludger Active, 'ABI': Alpha Bludger Inactive,
                # 'BS': Beta Seeker, 'BBA': Beta Bludger Active, 'BBI': Beta Bludger Inactive
                sector_occupancies[token_state.position] = {'AS': [], 'ABA': [], 'ABI': [],
                                                            'BS': [], 'BBA': [], 'BBI': []}
            if split_name[0] == "alpha":
                if split_name[1] == "seeker":
                    if token_state.satellite.fuel != self.kothgame.inargs.min_fuel:
                        sector_occupancies[token_state.position]['AS'] = [1]  # track active alpha seeker
                    else:
                        sector_occupancies[token_state.position]['AS'] = [-1]  # track inactive alpha seeker
                else:
                    if token_state.satellite.fuel != self.kothgame.inargs.min_fuel:
                        sector_occupancies[token_state.position]['ABA'].append(int(split_name[2]))  # track active alpha bludgers
                    else:
                        sector_occupancies[token_state.position]['ABI'].append(int(split_name[2]))  # track inactive alpha bludgers
            else:
                if split_name[1] == "seeker":
                    if token_state.satellite.fuel != self.kothgame.inargs.min_fuel:
                        sector_occupancies[token_state.position]['BS'] = [1]  # track active beta seeker
                    else:
                        sector_occupancies[token_state.position]['BS'] = [-1]  # track inactive beta seeker
                else:
                    if token_state.satellite.fuel != self.kothgame.inargs.min_fuel:
                        sector_occupancies[token_state.position]['BBA'].append(int(split_name[2]))  # track active beta bludgers
                    else:
                        sector_occupancies[token_state.position]['BBI'].append(int(split_name[2]))  # track inactive beta bludgers

        # for each occupied sector, draw all tokens within (Seekers depicted as squares, Bludgers depicted as circles)
        for sector, tokens in sector_occupancies.items():
            seeker_size = self._board_r / self._ring_count / 4
            bludger_size = 3 * seeker_size / 4
            # tracks total count and progress of drawing all level 1 tokens within the sector (Inactive Bludgers)
            token_count1 = len(tokens['ABI']) + len(tokens['BBI'])
            token_idx1 = 0
            # tracks total count and progress of drawing all level 2 tokens within the sector (Active Bludgers)
            token_count2 = len(tokens['ABA']) + len(tokens['BBA'])
            token_idx2 = 0
            # tracks total count and progress of drawing all level 3 tokens within the sector (Seekers)
            token_count3 = len(tokens['AS']) + len(tokens['BS'])
            token_idx3 = 0

            # iterate over and draw appropriate amount of each type of token within sector
            for token_type, token_ids in tokens.items():
                local_token_count = len(token_ids)
                if local_token_count:
                    # determine player / color
                    if token_type[0] == 'A':
                        color = self._p1_color
                        outline_color = self._p1_color_dark
                    else:
                        color = self._p2_color
                        outline_color = self._p2_color_dark

                    # determine token type (Seeker / Bludger)
                    if token_type[1] == 'S':
                        # determine center and angle of seeker
                        token_center, token_angle = self.get_token_coords(sector, token_count3, token_idx3, 3)
                        rect = (pol2cart(seeker_size, token_angle + 45, token_center),
                                pol2cart(seeker_size, token_angle + 135, token_center),
                                pol2cart(seeker_size, token_angle + 225, token_center),
                                pol2cart(seeker_size, token_angle + 315, token_center))
                        token_idx3 += 1
                        # determine whether active or inactive
                        if token_ids[0] == 1:
                            pg.draw.polygon(self._screen, color, rect)
                            pg.draw.polygon(self._screen, outline_color, rect, width=2)
                        elif token_ids[0] == -1:
                            pg.draw.polygon(self._screen, self._null_color, rect)
                            pg.draw.polygon(self._screen, outline_color, rect, width=2)
                        #Draw the token number - Always 0 for a seeker
                        self._screen.blit(self._large_font.render('0', True, (0, 0, 0)),
                            (token_center[0] - (0.5 * b_font_size[0]),
                            token_center[1] - (0.45 * b_font_size[1])))
                    else:
                        # determine whether active or inactive
                        if token_type[2] == 'A':
                            for idx, local_token_idx in enumerate(range(token_idx2, token_idx2 + local_token_count)):
                                token_center = self.get_token_coords(sector, token_count2, local_token_idx, 2)
                                pg.draw.circle(self._screen, color, token_center, bludger_size)
                                pg.draw.circle(self._screen, outline_color, token_center, bludger_size, width=2)
                                if token_ids[idx] != 10:
                                    self._screen.blit(self._large_font.render(str(token_ids[idx]), True, (0, 0, 0)),
                                                    (token_center[0] - (0.5 * b_font_size[0]),
                                                    token_center[1] - (0.45 * b_font_size[1])))
                                    token_idx2 += 1
                                else:
                                    self._screen.blit(self._large_font.render(str(token_ids[idx]), True, (0, 0, 0)),
                                                    (token_center[0] - (2.3 * b_font_size[0]),
                                                    token_center[1] - (0.45 * b_font_size[1])))
                                    token_idx2 += 1
                        else:
                            for idx, local_token_idx in enumerate(range(token_idx1, token_idx1 + local_token_count)):
                                token_center = self.get_token_coords(sector, token_count1, local_token_idx, 1)
                                pg.draw.circle(self._screen, self._null_color, token_center, bludger_size)
                                pg.draw.circle(self._screen, outline_color, token_center, bludger_size, width=2)
                                if token_ids[idx] != 10:
                                    self._screen.blit(self._large_font.render(str(token_ids[idx]), True, (0, 0, 0)),
                                                    (token_center[0] - (0.5 * b_font_size[0]),
                                                    token_center[1] - (0.45 * b_font_size[1])))
                                    token_idx1 += 1
                                else:
                                    self._screen.blit(self._large_font.render(str(token_ids[idx]), True, (0, 0, 0)),
                                                    (token_center[0] - (2.3 * b_font_size[0]),
                                                    token_center[1] - (0.45 * b_font_size[1])))
                                    token_idx1 += 1

    def get_token_coords(self, sector, token_count, token_idx, level):
        '''Takes a sector, a total number of tokens within the sector, a token index local to the sector, and a level,
        and retrieves the cartesian coordinates the token should be drawn at'''
        sector_ctr = 0
        ring = 0

        for ring_ctr in range(1, self._ring_count + 1):
            sector_ctr += 2 ** ring_ctr
            if sector_ctr >= sector:
                ring = ring_ctr
                break

        ring_sectors = 2 ** ring
        sector_rem = -(sector - (sector_ctr - ring_sectors))
        sector_init_angle = sector_rem * 360 / ring_sectors
        ring_r_min = int(ring / (self._ring_count + 1) * self._board_r)
        ring_r_max = int((ring + 1) / (self._ring_count + 1) * self._board_r)

        sector_mod_angle = (token_idx + 1) * 360 / ring_sectors / (token_count + 1)
        token_angle = sector_init_angle + sector_mod_angle

        if level == 1:
            token_r = ring_r_min + ((ring_r_max - ring_r_min) / 4)
        elif level == 2:
            token_r = ring_r_min + ((ring_r_max - ring_r_min) / 2)
        else:
            token_r = ring_r_min + (3 * (ring_r_max - ring_r_min) / 4)
            return pol2cart(token_r, sector_init_angle + sector_mod_angle, self._board_c), token_angle

        return pol2cart(token_r, token_angle, self._board_c)

    def _handle_events(self):
        '''
        Contains cycle that observes button presses and keystrokes to control program flow
        Buttons:
         - Play/Pause: plays program at desired speed or stops program; turns light green when pressed;
           also accessed by pressing space key.
         - Step: immediately advances to next phase; turns light yellow when pressed;
           also accessed by pressing right arrow key.
         - Speed Up/Slow Down: decrements or increments phase length, respectively; turn light aqua when pressed;
           also accessed by pressing up and down arrow keys, respectively.
         - Quit: quits pygame display and exits the program; turns light red when pressed;
           also accessed by pressing escape key.
         '''
        is_min_latency = (self._latency <= self._min_latency)

        # if button panel has not been initialized, initialize button panel
        if not self._buttons_active:
            self._button_panel = {'Play/Pause': RC.Button(self._screen, 'Play/Pause', self._is_paused, is_min_latency,
                                                          (self._x_dim - self._button_size,
                                                           (self._y_dim / 2) - (3 * self._button_size)),
                                                          self._button_size, self._null_color, self._title_color,
                                                          self._colors['light_green']),
                                  'Step': RC.Button(self._screen, 'Step', self._is_paused, is_min_latency,
                                                    (self._x_dim - self._button_size,
                                                     (self._y_dim / 2) - (2 * self._button_size)),
                                                    self._button_size, self._null_color, self._title_color,
                                                    self._colors['light_yellow']),
                                  'SpeedUp': RC.Button(self._screen, 'SpeedUp', self._is_paused, is_min_latency,
                                                       (self._x_dim - self._button_size,
                                                        (self._y_dim / 2) - (1 * self._button_size)),
                                                       self._button_size, self._null_color, self._title_color,
                                                       self._colors['light_aqua']),
                                  'SlowDown': RC.Button(self._screen, 'SlowDown', self._is_paused, is_min_latency,
                                                        (self._x_dim - self._button_size,
                                                         (self._y_dim / 2) + (1 * self._button_size)),
                                                        self._button_size, self._null_color, self._title_color,
                                                        self._colors['light_aqua']),
                                  'Quit': RC.Button(self._screen, 'Quit', self._is_paused, is_min_latency,
                                                    (self._x_dim - self._button_size,
                                                     (self._y_dim / 2) + (2 * self._button_size)),
                                                    self._button_size, self._null_color, self._title_color,
                                                    self._colors['light_red'])
                                  }
            self._buttons_active = True

        # draw each button in the button panel
        for button in self._button_panel.values():
            button.draw_button(False, self._is_paused, is_min_latency)

        # draw current phase length (latency) in seconds in middle of button panel
        self._draw_phase_length(self._x_dim - self._button_size, self._y_dim / 2)

        event_cycle = 0
        do_step = False
        do_quit = False

        # perform a cycle every 100 milliseconds to check for button presses and keystrokes until time
        # on the current phase has reached the desired length
        while event_cycle < self._latency:
            # self._draw_earth()  # Earth rotation advances each cycle

            for event in pg.event.get():
                if event.type == pg.MOUSEBUTTONDOWN:  # handle mouse clicks
                    # determine if any button was clicked
                    for button in self._button_panel.values():
                        if button.check_mouse_pos(pg.mouse.get_pos()):
                            # press the appropriate button
                            self._is_paused, self._latency, do_step, do_quit = \
                                button.press(self._is_paused, self._latency, self._min_latency)

                            # while playing, step function is unavailable
                            if button.type == 'Play' or button.type == 'Pause':
                                self._button_panel['Step'].draw_button(False, self._is_paused)
                            # update display of current phase length whenever it changes
                            elif button.type == 'SpeedUp' or button.type == 'SlowDown':
                                self._draw_phase_length(self._x_dim - self._button_size, self._y_dim / 2)

                elif event.type == pg.KEYDOWN:  # handle keystrokes
                    if event.key == pg.K_SPACE:  # space plays / pauses
                        self._is_paused, self._latency, do_step, do_quit = \
                            self._button_panel['Play/Pause'].press(self._is_paused, self._latency, self._min_latency)
                    elif event.key == pg.K_RIGHT:  # right arrow steps through phases
                        self._is_paused, self._latency, do_step, do_quit = \
                            self._button_panel['Step'].press(self._is_paused, self._latency, self._min_latency)
                    elif event.key == pg.K_UP:  # up arrow decrements latency and updates phase length display
                        self._is_paused, self._latency, do_step, do_quit = \
                            self._button_panel['SpeedUp'].press(self._is_paused, self._latency, self._min_latency)
                        self._draw_phase_length(self._x_dim - self._button_size, self._y_dim / 2)
                    elif event.key == pg.K_DOWN:  # down arrow increments latency and updates phase length display
                        self._is_paused, self._latency, do_step, do_quit = \
                            self._button_panel['SlowDown'].press(self._is_paused, self._latency, self._min_latency)
                        self._draw_phase_length(self._x_dim - self._button_size, self._y_dim / 2)
                    elif event.key == pg.K_ESCAPE:  # escape key quits pygame and exits the program
                        self._is_paused, self._latency, do_step, do_quit = \
                            self._button_panel['Quit'].press(self._is_paused, self._latency, self._min_latency)

            # continue to next phase
            if do_step:
                break

            # quit pygame and exit the program
            if do_quit:
                pg.quit()
                exit(0)

            if self._is_paused:
                event_cycle = 0  # pausing resets time on current phase
            else:
                event_cycle += 100  # increment by 100 milliseconds (length of wait)

            pg.time.wait(100)

    def _draw_phase_length(self, x, y):
        '''Display the current phase length (latency) in seconds within the button panel'''
        rect = pg.Rect(x, y, self._button_size, self._button_size)
        lighter = (self._null_color[0] + 50, self._null_color[1] + 50, self._null_color[2] + 50)
        darker = (self._null_color[0] - 50, self._null_color[1] - 50, self._null_color[2] - 50)

        pg.draw.rect(self._screen, self._bg_color, rect)  # cover up old phase length values
        # draw border
        pg.draw.line(self._screen, lighter, (x, y), (x + self._button_size, y), width=4)
        pg.draw.line(self._screen, lighter, (x, y), (x, y + self._button_size), width=4)
        pg.draw.line(self._screen, darker, (x + self._button_size, y),
                     (x + self._button_size, y + self._button_size), width=4)
        pg.draw.line(self._screen, darker, (x, y + self._button_size),
                     (x + self._button_size, y + self._button_size), width=4)

        # pl_title = self._font_bold.render("Phase", False, self._title_color)
        # self._screen.blit(pl_title,
        #                   (x + (5.5 * self._font_bold.size(' ')[0]), y + (0.5 * self._font_bold.size(' ')[1])))
        # pl_title = self._font_bold.render("Length", False, self._title_color)
        # self._screen.blit(pl_title,
        #                   (x + (4.7 * self._font_bold.size(' ')[0]), y + (1.5 * self._font_bold.size(' ')[1])))

        # draw phase length value
        pl_str = str(self._latency / 1000)
        pl_text = self._large_font.render(pl_str, True, self._colors['light_aqua'])
        self._screen.blit(pl_text, (x + (self._button_size / 2) - (len(pl_str) * self._large_font.size(' ')[0]),
                                    (y + (self._button_size / 2) - (0.5 * self._large_font.size(' ')[1]))))

        # update only the area of the button
        pg.display.update(rect)

    def _draw_earth(self):
        '''
        Displays image of the Earth in the center of the game board that rotates by 180 degrees at each drift phase.
        '''
        # Earth only rotates on drift phase
        if self.kothgame.game_state[U.TURN_PHASE] == "movement" and self.kothgame.game_state[U.TURN_COUNT] > 0:
            cycles = 9
            rot_increment = 20
        else:
            cycles = 1
            rot_increment = 0

        ring_r_min = self._board_r / (self._ring_count + 1)
        for rot_frame in range(cycles):
            # Increment rotation
            self._earth_rotation += rot_increment

            # load the image
            earth_img = pg.image.load(Path(__file__).parent.joinpath("earth.png"))
            earth_img = pg.transform.scale(earth_img, (2 * ring_r_min, 2 * ring_r_min))

            # offset from pivot to center
            w, h = earth_img.get_size()
            img_rect = earth_img.get_rect(topleft=(self._board_c[0] - (w / 2), self._board_c[1] - (h / 2)))
            offset_center_to_pivot = pg.math.Vector2(self._board_c) - img_rect.center

            # rotated offset from pivot to center
            rotated_offset = offset_center_to_pivot.rotate(-self._earth_rotation)

            # get rotated image center
            rotated_image_center = (self._board_c[0] - rotated_offset.x, self._board_c[1] - rotated_offset.y)

            # rotate the image and its rectangle
            rot_img = pg.transform.rotate(earth_img, self._earth_rotation)
            rot_img_rect = rot_img.get_rect(center=rotated_image_center)

            # blit the rotated image
            self._screen.blit(rot_img, rot_img_rect)

            # redraw board lines that get covered by rotated image
            pg.draw.aaline(self._screen, self._board_color, (self._board_c[0] - ring_r_min, self._board_c[1]),
                           (self._board_c[0] - (2 * ring_r_min), self._board_c[1]))
            pg.draw.aaline(self._screen, self._board_color, (self._board_c[0] + ring_r_min, self._board_c[1]),
                           (self._board_c[0] + (2 * ring_r_min), self._board_c[1]))

            pg.display.update(rot_img_rect)

    def draw_win(self, winner):
        '''Displays winner clearly and updates score when game finishes'''
        if winner == "alpha":
            win_color = self._p1_color
        elif winner == "beta":
            win_color = self._p2_color
        else:
            win_color = self._null_color

        vl_font_size = self._very_large_font_bold.size(' ')
        lb_font_size = self._large_font_bold.size(' ')
        l_font_size = self._large_font.size(' ')
        winner_title = self._very_large_font_bold.render("Winner: ", True, self._title_color)
        self._screen.blit(winner_title, ((self._x_dim / 2) - (5 * vl_font_size[0]), self._margins[1]))
        winner_text = self._very_large_font_bold.render(winner.capitalize(), True, win_color)
        self._screen.blit(winner_text, ((self._x_dim / 2) + (12 * vl_font_size[0]), self._margins[1]))

        # display score
        l_score_len = len(str(self.kothgame.game_state[U.P1][U.SCORE]))
        score_rect = ((0.75 * self._x_dim) - ((l_score_len + 4) * l_font_size[0]),  # left
                      self._margins[1],  # top
                      20 * lb_font_size[0],  # width
                      2 * lb_font_size[1])  # height
        pg.draw.rect(self._screen, self._bg_color, score_rect)

        score_title = self._large_font_bold.render("Score:", True, self._title_color)
        self._screen.blit(score_title, ((0.75 * self._x_dim) - (3 * lb_font_size[0]), self._margins[1]))

        divider = self._large_font_bold.render("|", True, self._title_color)
        self._screen.blit(divider, ((0.75 * self._x_dim) + (3 * lb_font_size[0]),
                                    self._margins[1] + lb_font_size[1]))

        p1_score = self._large_font.render(str(self.kothgame.game_state[U.P1][U.SCORE]), True, self._p1_color)
        self._screen.blit(p1_score, ((0.75 * self._x_dim) - ((l_score_len + 2) * l_font_size[0]),
                                     self._margins[1] + lb_font_size[1]))

        p2_score = self._large_font.render(str(self.kothgame.game_state[U.P2][U.SCORE]), True, self._p2_color)
        self._screen.blit(p2_score, ((0.75 * self._x_dim) + (6 * l_font_size[0]),
                                     self._margins[1] + lb_font_size[1]))

        pg.display.update()

        if self._render_mode == "human":
            pg.time.wait(self._latency)
        elif self._render_mode == "debug":
            self._is_paused = True
            self._handle_events()

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''

        # deactivates pygame library and closes render window
        pg.time.wait(self._latency)
        #pg.close()
        #pg.quit() #Don't need either, let the user decide when to close the window

    def reset(self):
        '''
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        '''
        self.agents = self.possible_agents[:]
        self.kothgame.reset_game()
        observations = self.encode_all_observaitons()
        self.legal_actions = {agent: observations[agent]['action_mask'] for agent in observations.keys()}
        self.gameover = False
        self.render_json = None
        return observations

    def step(self, actions):
        '''
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        '''
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            print('No actions passed into step()')
            self.agents = []
            return {}, {}, {}, {}

        # convert gym-encoded actions to verbose actions and pass to koth game
        self.verbose_actions = self.decode_all_discrete_actions(actions=actions)
        verbose_actions = self.decode_all_discrete_actions(actions=actions)

        # Update state of game
        if self.kothgame.game_state[U.TURN_PHASE] == U.MOVEMENT:
            # apply movement actions to progress to engagement
            rewards = self.kothgame.apply_verbose_actions(actions=verbose_actions)

            if self.render_json is not None:
                # gs_dict = U.get_game_state(self.kothgame.game_state, self.kothgame.token_catalog)
                gs_formatted = UJD.format_response_message(GS.MOVE_PHASE_RESP, GS.MOVE_PHASE, self.kothgame)
                json.dump(gs_formatted, self.render_json)
                self.render_json.write('\n')
                print(f"Wrote {GS.MOVE_PHASE_RESP} message")   

        elif self.kothgame.game_state[U.TURN_PHASE] == U.ENGAGEMENT:

            if self.render_json is not None:
                # gs_dict = U.get_game_state(self.kothgame.game_state, self.kothgame.token_catalog)
                acts_formatted = UJD.format_action_message(GS.ENGAGE_PHASE_REQ, GS.ENGAGE_PHASE, self.kothgame, verbose_actions)
                json.dump(acts_formatted, self.render_json)
                self.render_json.write('\n')
                print(f"Wrote {GS.ENGAGE_PHASE_REQ} message")

            # apply engagement actions to progress to engagment
            eng_rew = self.kothgame.apply_verbose_actions(actions=verbose_actions)
            #assert self.kothgame.game_state[U.TURN_PHASE] == U.DRIFT

            if self.render_json is not None:
                # gs_dict = U.get_game_state(self.kothgame.game_state, self.kothgame.token_catalog)
                gs_formatted = UJD.format_response_message(GS.ENGAGE_PHASE_RESP, GS.ENGAGE_PHASE, self.kothgame)
                json.dump(gs_formatted, self.render_json)
                self.render_json.write('\n')
                print(f"Wrote {GS.ENGAGE_PHASE_RESP} message")

            assert self.kothgame.game_state[U.TURN_PHASE] == U.DRIFT
            
            # immediately step through drift phase since no actions available
            drf_rew = self.kothgame.apply_verbose_actions(actions=None)

            if self.render_json is not None:
                # gs_dict = U.get_game_state(self.kothgame.game_state, self.kothgame.token_catalog)
                gs_formatted = UJD.format_response_message(GS.DRIFT_PHASE_RESP, GS.DRIFT_PHASE, self.kothgame)
                json.dump(gs_formatted, self.render_json)
                self.render_json.write('\n')
                print(f"Wrote {GS.DRIFT_PHASE_RESP} message")
            
            # combine rewards:
            rewards = {ag_id: eng_rew[ag_id] + drf_rew[ag_id] for ag_id in self.agents}

        else:
            raise ValueError('Unexpected turn phase: {}.\n' +
                             'Note: drift phase should be automatically stepped through after engagement ' +
                             'and not encountered here')

        # generate new observations from game state
        observations = self.encode_all_observaitons()

        # update legal actions for next game step
        self.legal_actions = {agent: observations[agent]['action_mask'] for agent in observations.keys()}

        # Check for game termination
        dones = {agent: self.kothgame.game_state[U.GAME_DONE] for agent in self.agents}

        if all(dones.values()):
            self.gameover = True
            #print('Game has ended')
            
         #Close file if game is over
        if self.gameover and (self.render_json is not None):
            self.render_json.close()
            print('Closing json file')
            
        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, dones, infos

    def encode_all_observaitons(self):
        '''return gym-encodedd observation and action mask for each player'''

        return {agent: {
            'observation': self.encode_player_observation(player_id=agent)[0],
            'action_mask': self.encode_legal_action_mask(player_id=agent)
        } for agent in self.agents}

    def decode_all_flat_actions(self, actions):
        '''return decoded, verbose actions for all players'''

        # return None if actions are None
        # important for drift phase
        if actions is None:
            return None

        verbose_actions = {}
        for plr_id, plr_act in actions.items():
            verbose_actions.update(self.decode_flat_player_action(player_id=plr_id, player_act=plr_act))
        return verbose_actions

    def decode_all_discrete_actions(self, actions):
        '''return decoded, verbose actions for all players from tuple of discrete actions'''

        # return None if actions are None
        # important for drift phase
        if actions is None:
            return None

        verbose_actions = {}
        for plr_id, plr_act in actions.items():
            verbose_actions.update(self.decode_discrete_player_action(player_id=plr_id, player_act=plr_act))
        return verbose_actions

    def encode_all_flat_actions(self, actions):
        '''return gym-encoded actions for each player from dict of verbose actions'''

        # return None if actions are None
        # important for drift phase
        if actions is None:
            return None

        sep_actions = dict()
        for plr_id in self.agents:
            sep_actions[plr_id] = {tok_id: tok_act for tok_id, tok_act in
                                   actions.items() if koth.parse_token_id(tok_id)[0] == plr_id}

        return {ag_id: self.encode_flat_player_action(ag_act) for ag_id, ag_act in sep_actions.items()}

    def encode_all_discrete_actions(self, actions):
        '''return gym-encoded actions for each player from dict of verbose actions'''

        # return None if actions are None
        # important for drift phase
        if actions is None:
            return None

        sep_actions = dict()
        for plr_id in self.agents:
            sep_actions[plr_id] = {tok_id: tok_act for tok_id, tok_act in
                                   actions.items() if koth.parse_token_id(tok_id)[0] == plr_id}

        return {ag_id: self.encode_discrete_player_action(ag_act) for ag_id, ag_act in sep_actions.items()}

    def encode_random_valid_discrete_actions(self):
        '''return random valid actions encoded in flat gym space'''
        rand_acts = self.kothgame.get_random_valid_actions()
        return self.encode_all_discrete_actions(rand_acts)

    def encode_token_observation(self, player_id: str, token_id: str):
        ''' encode a single token's state for flat and non-flat gym observations

        Args:
            player_id (str): which player making the observation
            token_id (str): string identifier of token

        Returns: 
            flat_obs : ndarray(shape=(N_BITS_OBS_PER_TOKEN,))
                observation of single token flattened into flat_per_token obs space
            tuple_obs : tuple
                observation of single token (own_piece, role, position, fuel, ammo)
        Notes:
            - Normalizes all entity information to range [0,1]
            - Loosely based AlphaStar entity_list encoding. See: 
            https://github.com/mit-ll/spacegym-od2d/wiki/alphastar-detailed-architecture.txt 
            - Also compare to OpenAI Five encoding which doesn't use one-hots, but does normalize
            https://arxiv.org/abs/1912.06680
            own_piece : ndarray(shape=(N_BITS_OWN_PIECE,))
                boolean whether it's your piece (1) or opponents (0)
            role : ndarray(shape=(N_BITS_ROLE,)) 
                role of piece (seeker=0, bludger=1) represented as one-hot
            position : ndarray(shape=(N_BITS_POSITION,))
                sector number of location of piece on board represented as one-hot
            fuel : ndarray(shape=(N_BITS_FUEL,))
                binary representation of fuel remaining, rounded down to nearest int
            ammo : ndarray(shape=(N_BITS_AMMO,))
                binary representation of ammo remaining  
        '''
        # obs = dict()
        # check valid token and player
        assert player_id in [U.P1, U.P2]
        tok_p, tok_r, tok_n = koth.parse_token_id(token_id)
        assert tok_p in [U.P1, U.P2]
        # assert tok_p == player_id

        # encode own_piece, verify against gym_space
        obs_own_piece = [bool(tok_p == player_id)]
        flat_obs_own_piece = spaces.flatten(self.obs_space_info.per_token_components.own_piece, obs_own_piece)
        assert self.obs_space_info.per_token_components.own_piece.contains(obs_own_piece)
        assert self.obs_space_info.flat_per_token_components.own_piece.contains(flat_obs_own_piece)

        # encode role, verify against obs_space_info
        assert tok_r == self.kothgame.token_catalog[token_id].role
        obs_role = U.PIECE_ROLES.index(tok_r)
        flat_obs_role = spaces.flatten(self.obs_space_info.per_token_components.role, obs_role)
        assert self.obs_space_info.per_token_components.role.contains(obs_role)
        assert self.obs_space_info.flat_per_token_components.role.contains(flat_obs_role)

        # encode position, verify against obs_space_info
        obs_position = self.kothgame.token_catalog[token_id].position
        flat_obs_position = spaces.flatten(self.obs_space_info.per_token_components.position, obs_position)
        assert self.obs_space_info.per_token_components.position.contains(obs_position)
        assert self.obs_space_info.flat_per_token_components.position.contains(flat_obs_position)

        # encode fuel, verify against obs_space_info
        if self.kothgame.token_catalog[token_id].satellite.fuel < 0:
            tmp_fuel = 0
        else:
            tmp_fuel = self.kothgame.token_catalog[token_id].satellite.fuel
        obs_fuel = get_non_negative_binary_observation(tmp_fuel, N_BITS_OBS_FUEL)
        flat_obs_fuel = spaces.flatten(self.obs_space_info.per_token_components.fuel, obs_fuel)
        assert self.obs_space_info.per_token_components.fuel.contains(obs_fuel), "obs_fuel: {}".format(obs_fuel)
        assert self.obs_space_info.flat_per_token_components.fuel.contains(flat_obs_fuel)

        # encode ammo, verfiy against obs_space_info
        obs_ammo = get_non_negative_binary_observation(self.kothgame.token_catalog[token_id].satellite.ammo,
                                                       N_BITS_OBS_AMMO)
        flat_obs_ammo = spaces.flatten(self.obs_space_info.per_token_components.ammo, obs_ammo)
        assert self.obs_space_info.per_token_components.ammo.contains(obs_ammo)
        assert self.obs_space_info.flat_per_token_components.ammo.contains(flat_obs_ammo)

        # concatentate flat observations
        flat_obs = np.concatenate(
            (flat_obs_own_piece, flat_obs_role, flat_obs_position, flat_obs_fuel, flat_obs_ammo),
            dtype=self.obs_space_info.flat_per_token.dtype)
        assert self.obs_space_info.flat_per_token.contains(flat_obs)
        tuple_obs = (obs_own_piece, obs_role, obs_position, obs_fuel, obs_ammo)
        return flat_obs, tuple_obs

    def encode_scoreboard_observation(self, player_id: str):
        '''encode scoreboard observation from perspective of a specific player into flat 1D gym space
        Args:
            player_id : (str) 
                which player making the observation

        Returns:
            flat_obs : ndarray(shape=(N_BITS_OBS_PER_TOKEN,))
                observation of single token flattened into flat_per_token obs space
            dict_obs : tuple
                observation of single token
        '''

        if player_id == U.P1:
            own_hill = U.GOAL1
            opp_hill = U.GOAL2
            opp_id = U.P2
        elif player_id == U.P2:
            own_hill = U.GOAL2
            opp_hill = U.GOAL1
            opp_id = U.P1
        else:
            raise ValueError("Unrecognized player_id: {}".format(player_id))

        # turn_phase observation
        obs_turn_phase = U.TURN_PHASE_LIST.index(self.kothgame.game_state[U.TURN_PHASE])
        assert self.obs_space_info.scoreboard['turn_phase'].contains(obs_turn_phase)
        flat_obs_turn_phase = spaces.flatten(self.obs_space_info.scoreboard['turn_phase'], obs_turn_phase)
        assert flat_obs_turn_phase.shape == (N_BITS_OBS_TURN_PHASE,)

        # turn_count observation
        obs_turn_count = get_non_negative_binary_observation(self.kothgame.game_state[U.TURN_COUNT],
                                                             N_BITS_OBS_TURN_COUNT)
        assert self.obs_space_info.scoreboard['turn_count'].contains(obs_turn_count)
        flat_obs_turn_count = spaces.flatten(self.obs_space_info.scoreboard['turn_count'], obs_turn_count)
        assert flat_obs_turn_count.shape == (N_BITS_OBS_TURN_COUNT,)

        # own-score observation
        obs_own_score = get_binary_observation(self.kothgame.game_state[player_id][U.SCORE], N_BITS_OBS_SCORE - 1)
        obs_own_score = tuple((np.array([True]), obs_own_score))
        assert self.obs_space_info.scoreboard['own_score'].contains(obs_own_score)
        flat_obs_own_score = spaces.flatten(self.obs_space_info.scoreboard['own_score'], obs_own_score)
        assert flat_obs_own_score.shape == (N_BITS_OBS_SCORE,)

        # opponent-score observation
        obs_opp_score = get_binary_observation(self.kothgame.game_state[opp_id][U.SCORE], N_BITS_OBS_SCORE - 1)
        obs_opp_score = tuple((np.array([False]), obs_opp_score))
        assert self.obs_space_info.scoreboard['opponent_score'].contains(obs_opp_score)
        flat_obs_opp_score = spaces.flatten(self.obs_space_info.scoreboard['opponent_score'], obs_opp_score)
        assert flat_obs_opp_score.shape == (N_BITS_OBS_SCORE,)

        # own-hill observation
        obs_own_hill = self.kothgame.game_state[own_hill]
        obs_own_hill = tuple((np.array([True]), obs_own_hill))
        assert self.obs_space_info.scoreboard['own_hill'].contains(obs_own_hill) #TODO: Figure out why I can't get this to work
        flat_obs_own_hill = spaces.flatten(self.obs_space_info.scoreboard['own_hill'], obs_own_hill)
        assert flat_obs_own_hill.shape == (N_BITS_OBS_HILL,)

        # own-hill observation
        obs_opp_hill = self.kothgame.game_state[opp_hill]
        obs_opp_hill = tuple((np.array([False]), obs_opp_hill))
        assert self.obs_space_info.scoreboard['own_hill'].contains(obs_opp_hill)
        flat_obs_opp_hill = spaces.flatten(self.obs_space_info.scoreboard['own_hill'], obs_opp_hill)
        assert flat_obs_opp_hill.shape == (N_BITS_OBS_HILL,)

        flat_obs = np.concatenate(
            (flat_obs_turn_phase,
             flat_obs_turn_count,
             flat_obs_own_score,
             flat_obs_opp_score,
             flat_obs_own_hill,
             flat_obs_opp_hill), dtype=self.obs_space_info.flat_scoreboard.dtype)
        assert self.obs_space_info.flat_scoreboard.contains(flat_obs)

        dict_obs = OrderedDict({
            'turn_phase': obs_turn_phase,
            'turn_count': obs_turn_count,
            'own_score': obs_own_score,
            'opponent_score': obs_opp_score,
            'own_hill': obs_own_hill,
            'opponent_hill': obs_opp_hill})

        return flat_obs, dict_obs

    def encode_legal_action_mask(self, player_id: str):
        '''encode the legal actions at current game state into 2D gym Box space'''

        # check valid player id
        assert player_id in self.agents

        # get verbose legal actions from game state
        legal_acts = self.kothgame.game_state[U.LEGAL_ACTIONS]

        # init legal_act_mask as 2D, reshape to 1D later
        encoded_legal_acts_mask = np.zeros(
            (self.n_tokens_per_player, N_BITS_ACT_PER_TOKEN),
            dtype=self.act_space_info.flat_per_token.dtype)

        # iterate through each token in legal_acts
        for tok_id, tok_legacts in legal_acts.items():
            tok_plr, _, tok_num = koth.parse_token_id(tok_id)
            tok_num = int(tok_num)

            # only analyze queried player
            if tok_plr != player_id:
                continue

            # iterate through each legal action, combining
            for legact in tok_legacts:
                encoded_legal_acts_mask[tok_num] = (
                        encoded_legal_acts_mask[tok_num] |
                        self.encode_flat_token_action(legact).astype(self.act_space_info.flat_per_token.dtype)
                )

        return encoded_legal_acts_mask
        # reshape legal_acts_mask and return
        # return encoded_legal_acts_mask.reshape((N_BITS_ACT_PER_PLAYER,))

    def decode_legal_action_mask(self, player_id, legal_acts_mask):
        '''convert flat 2D gym Box array of legal actions into dictionary'''

        # assert legal_acts_mask.shape == (N_BITS_ACT_PER_PLAYER,)
        assert legal_acts_mask.shape == (self.n_tokens_per_player, N_BITS_ACT_PER_TOKEN)

        # # reshape to 2D for ease of handling
        # leg_acts_mask = legal_acts_mask.reshape(self.n_tokens_per_player, N_BITS_ACT_PER_TOKEN)

        decoded_legal_acts = dict()
        for tok_num, tok_legacts in enumerate(legal_acts_mask):
            tok_id = self.kothgame.get_token_id(player_id=player_id, token_num=tok_num)
            decoded_legal_acts[tok_id] = []
            for disc_act in np.nonzero(tok_legacts)[0]:
                encoded_act = np.zeros((N_BITS_ACT_PER_TOKEN,), dtype=self.act_space_info.flat_per_token.dtype)
                encoded_act[disc_act] = 1
                decoded_legal_acts[tok_id].append(self.decode_flat_token_action(tok_id, encoded_act))

        return decoded_legal_acts

    def encode_player_observation(self, player_id: str):
        ''' encode a player's perspective of the board state into a flattened 1D gym space
        Args:
            player_id : (str) 
                which player making the observation

        Returns:
            flat_obs : ndarray(shape=(N_BITS_OBS_PER_TOKEN,))
                observation of single token flattened into flat_per_token obs space
            tuple_obs : tuple
                observation of single token (own_piece, role, position, fuel, ammo)

        Notes:
            - Loosely based AlphaStar entity_list encoding. See: 
            https://github.com/mit-ll/spacegym-od2d/wiki/alphastar-detailed-architecture.txt
        '''

        assert player_id in self.agents
        # get scoreboard information
        flat_obs_scoreboard, dict_obs_scoreboard = self.encode_scoreboard_observation(player_id=player_id)

        # get own-token observations
        own_tokens = [self.encode_token_observation(player_id=player_id, token_id=tok) for
                      tok in self.kothgame.token_catalog.keys() if koth.parse_token_id(tok)[0] == player_id]
        tuple_obs_own_tokens = tuple([ot[1] for ot in own_tokens])
        flat_obs_own_tokens = np.concatenate([ot[0] for ot in own_tokens])

        # get opponent-token observations
        opp_tokens = tuple(
            [self.encode_token_observation(player_id=player_id, token_id=tok) for
             tok in self.kothgame.token_catalog.keys() if koth.parse_token_id(tok)[0] != player_id])
        tuple_obs_opp_tokens = tuple([ot[1] for ot in opp_tokens])
        flat_obs_opp_tokens = np.concatenate([ot[0] for ot in opp_tokens])

        # format into dictionary observation
        dict_obs = OrderedDict({
            'scoreboard': dict_obs_scoreboard,
            'own_tokens': tuple_obs_own_tokens,
            'opponent_tokens': tuple_obs_opp_tokens
        })
        self.obs_space_info.per_player.contains(dict_obs)

        # flatten
        # flat_obs = spaces.flatten(self.obs_space_info.per_player, dict_obs)
        flat_obs = np.concatenate((flat_obs_scoreboard, flat_obs_own_tokens, flat_obs_opp_tokens))
        assert flat_obs.shape == (N_BITS_OBS_PER_PLAYER,)
        assert self.obs_space_info.flat_per_player.contains(flat_obs)

        return flat_obs, dict_obs

    def decode_flat_player_action(self, player_id, player_act):
        '''convert a player's flat gym action vector to koth verbose action dictionary
        
        Args:
            player_id : str
                string identifier for player taking action
            player_act : ndarray(shape=(N_BITS_ACT_PER_PLAYER,)
                flatten action vector of player

        Returns
            decoded_act : dict
                verbose action description
                key is piece id token_catalog, one for each piece in game
                value is the piece's movement tuple (U.MOVEMENT_TYPES) or 
                engagement tuple (ENGAGMENT_TYPE, target_piece_id, prob)

        '''

        # check player is valid
        assert player_id in self.agents

        # check action is valid in action space
        assert self.act_space_info.flat_per_player.contains(player_act), \
            "Action space {} does not contain action {} of shape {}".format(
                self.act_space_info.flat_per_player, player_act, player_act.shape
            )

        # init decoded action dictionary
        decoded_act = dict()

        # iterate through each token slice 
        for tok_num, tok_act in enumerate(player_act.reshape(self.n_tokens_per_player, N_BITS_ACT_PER_TOKEN)):
            # get token id
            tok_id = self.kothgame.get_token_id(player_id=player_id, token_num=tok_num)

            # decode token action
            decoded_act[tok_id] = self.decode_flat_token_action(token_id=tok_id, token_act=tok_act)

        return decoded_act

    def decode_discrete_player_action(self, player_id, player_act):
        '''convert a player's tuple of discrete gym action to koth verbose action dictionary
        
        Args:
            player_id : str
                string identifier for player taking action
            player_act : tuple(len=n_tokens_per_player)
                tuple of discrete actions for each token

        Returns
            decoded_act : dict
                verbose action description
                key is piece id token_catalog, one for each piece in game
                value is the piece's movement tuple (U.MOVEMENT_TYPES) or 
                engagement tuple (ENGAGMENT_TYPE, target_piece_id, prob)

        '''

        # check player is valid
        assert player_id in self.agents

        # check action is valid in action space
        assert self.act_space_info.per_player.contains(player_act), \
            "Action space {} does not contain action {} of shape {}".format(
                self.act_space_info.per_player, player_act, player_act.shape
            )

        # init decoded action dictionary
        decoded_act = dict()

        # iterate through each token slice 
        for tok_num, disc_tok_act in enumerate(player_act):
            # get token id
            tok_id = self.kothgame.get_token_id(player_id=player_id, token_num=tok_num)

            # decode token action
            onehot_tok_act = np.zeros(N_BITS_ACT_PER_TOKEN, dtype=self.act_space_info.flat_per_token.dtype)
            onehot_tok_act[disc_tok_act] = 1
            decoded_act[tok_id] = self.decode_discrete_token_action(token_id=tok_id, token_act=disc_tok_act)

        return decoded_act

    def encode_flat_player_action(self, player_act):
        '''convert a player's koth verbose action dictionary to flat gym action vector 
        
        Args:
            player_act : dict
                verbose action description
                key is piece id token_catalog, one for each piece in game
                value is the piece's movement tuple (U.MOVEMENT_TYPES) or 
                engagement tuple (ENGAGMENT_TYPE, target_piece_id, prob)

        Returns
            encoded_act : ndarray(shape=(N_BITS_ACT_PER_PLAYER,)
                flatten action vector of player

        '''

        # verify that action space dtypes are consistent to ensure no
        # unintentional truncation of values
        assert self.act_space_info.flat_per_player.dtype == self.act_space_info.flat_per_token.dtype

        # init encoded act as 2D array and reshape to 1D later
        encoded_act = np.zeros((self.n_tokens_per_player, N_BITS_ACT_PER_TOKEN),
                               dtype=self.act_space_info.flat_per_player.dtype)

        for tok_id, tok_act in player_act.items():
            _, _, tok_num = koth.parse_token_id(tok_id)
            tok_num = int(tok_num)
            encoded_act[tok_num] = self.encode_flat_token_action(token_act=tok_act)

        return encoded_act.reshape((N_BITS_ACT_PER_PLAYER,))

    def encode_discrete_player_action(self, player_act):
        '''convert a player's koth verbose action dictionary to tuple of discrete gym actions
        
        Args:
            player_act : dict
                verbose action description
                key is piece id token_catalog, one for each piece in game
                value is the piece's movement tuple (U.MOVEMENT_TYPES) or 
                engagement tuple (ENGAGMENT_TYPE, target_piece_id, prob)

        Returns
            encoded_act : tuple(len=N_BITS_ACT_PER_PLAYER)
                tuple of discrete actions for each token

        '''

        # init encoded act 1D array
        encoded_act = np.zeros(self.n_tokens_per_player,
                               dtype=self.act_space_info.flat_per_token.dtype)

        for tok_id, tok_act in player_act.items():
            _, _, tok_num = koth.parse_token_id(tok_id)
            tok_num = int(tok_num)
            encoded_act[tok_num] = self.encode_discrete_token_action(token_act=tok_act)

        return tuple(encoded_act)

    def decode_flat_token_action(self, token_id, token_act):
        '''convert a token's flat gym action vector into a koth Movement or EngagementTuple
        
        Args:
            token_id : (str) 
                which token taking the action
            token_act : ndarray(shape=(N_BITS_ACT_PER_TOKEN,))
                One-hot vector that encodes a token's action. ordering:
                0..4:movements, 5..4+N:shoot_attack, 5+N..4+2N:collide_attack, 5+2N..4+3N:guard

        Returns:
            action tuple (MovementTuple or EngagementTuple)
        '''

        # check action is valid in action space
        assert self.act_space_info.flat_per_token.contains(token_act), "Unrecognized action {}".format(token_act)

        # convert from one-hot to discrete
        # if token action is not a one-hot, then it is an invalid action
        # return invalid action flag as decoded action
        disc_act = np.nonzero(token_act)[0]
        if len(disc_act) != 1:
            # raise ValueError("Unexpected action discretization: {}".format(disc_act))
            return U.INVALID_ACTION
        disc_act = disc_act[0]

        return self.decode_discrete_token_action(token_id=token_id, token_act=disc_act)

    def decode_discrete_token_action(self, token_id, token_act):
        '''convert a token's flat gym action vector into a koth Movement or EngagementTuple
        
        Args:
            token_id : (str) 
                which token taking the action
            token_act : (int)
                0..4:movements, 5..4+N:shoot_attack, 5+N..4+2N:collide_attack, 5+2N..4+3N:guard

        Returns:
            action tuple (MovementTuple or EngagementTuple)
        '''

        # parse out token id
        player_id, _, _ = koth.parse_token_id(token_id)

        # get opponent id
        if player_id == self.agents[0]:
            opp_id = self.agents[1]
        elif player_id == self.agents[1]:
            opp_id = self.agents[0]
        else:
            raise ValueError("Unrecognized player_id: {}".format(player_id))

        # decode discrete action
        act_type = None
        targ_num = None
        engage_prob = None
        decoded_act = None

        if token_act == 0:
            # NOOP
            act_type = U.NOOP
            if self.kothgame.game_state[U.TURN_PHASE] == U.MOVEMENT:
                decoded_act = U.MovementTuple(action_type=act_type)
            elif self.kothgame.game_state[U.TURN_PHASE] == U.ENGAGEMENT:
                decoded_act = U.EngagementTuple(
                    action_type=act_type,
                    target=token_id,
                    prob=self.kothgame.inargs.engage_probs[U.P1][U.IN_SEC][U.NOOP]) #Technically should check p1 or p2, but NOOP will always be 1.0 prop anyways.
            else:
                raise ValueError("Unexpected turn phase: {}".format(self.kothgame.game_state[U.TURN_PHASE]))

        elif token_act in range(*self.act_space_info.move_slice):
            # Movement type action
            act_type = U.MOVEMENT_TYPES[token_act]
            decoded_act = U.MovementTuple(action_type=act_type)

        else:
            # engagement action

            if token_act in range(*self.act_space_info.shoot_slice):
                # Shoot type action
                act_type = U.SHOOT
                targ_num = token_act - self.act_space_info.shoot_slice[0]

                # parse target id (always opponent player)
                targ_id = self.kothgame.get_token_id(player_id=opp_id, token_num=targ_num)

            elif token_act in range(*self.act_space_info.collide_slice):
                # collide type action
                act_type = U.COLLIDE
                targ_num = token_act - self.act_space_info.collide_slice[0]

                # parse target id (always opponent player)
                targ_id = self.kothgame.get_token_id(player_id=opp_id, token_num=targ_num)

            elif token_act in range(*self.act_space_info.guard_slice):
                # Guard type action
                act_type = U.GUARD
                targ_num = token_act - self.act_space_info.guard_slice[0]

                # parse target id (always own player)
                targ_id = self.kothgame.get_token_id(player_id=player_id, token_num=targ_num)

            else:
                raise ValueError("Unexpected Action {}. Expected value in range 0-{}".format(
                    token_act, self.act_space_info.guard_slcide[1]
                ))

            # compute engagement probability
            engage_prob = self.kothgame.get_engagement_probability(
                token_id=token_id,
                target_id=targ_id,
                engagement_type=act_type)

            # compose engagement tuple
            decoded_act = U.EngagementTuple(action_type=act_type, target=targ_id, prob=engage_prob)

        return decoded_act

    def encode_flat_token_action(self, token_act):
        '''convert a verbose action tuple into flat gym action space
        
        Args:
            token_id : (str) 
                which token taking the action
            token_act : action tuple (MovementTuple or EngagementTuple)

        Returns:
            ndarray(shape=(N_BITS_ACT_PER_TOKEN,))
                One-hot vector that encodes a token's action. ordering:
                0..4:movements, 5..4+N:shoot_attack, 5+N..4+2N:collide_attack, 5+2N..4+3N:guard

        '''

        encoded_act = np.zeros((N_BITS_ACT_PER_TOKEN,),
                               dtype=self.act_space_info.flat_per_token.dtype)

        if token_act.action_type == U.NOOP:
            encoded_act[0] = 1
        elif isinstance(token_act, U.MovementTuple):
            encoded_act[U.MOVEMENT_TYPES.index(token_act.action_type)] = 1
        elif isinstance(token_act, U.EngagementTuple):

            # parse target number
            _, _, targ_num = koth.parse_token_id(token_act.target)
            targ_num = int(targ_num)

            if token_act.action_type == U.SHOOT:
                encoded_act[self.act_space_info.shoot_slice[0] + targ_num] = 1
            elif token_act.action_type == U.COLLIDE:
                encoded_act[self.act_space_info.collide_slice[0] + targ_num] = 1
            elif token_act.action_type == U.GUARD:
                encoded_act[self.act_space_info.guard_slice[0] + targ_num] = 1
            else:
                raise ValueError('Unrecognized action_type: {}'.format(token_act.action_type))

        else:
            raise ValueError('Unexpected token action type {}'.format(token_act))

        return encoded_act

    def encode_discrete_token_action(self, token_act):
        '''convert a verbose action tuple into discrete gym action space
        
        Args:
            token_id : (str) 
                which token taking the action
            token_act : action tuple (MovementTuple or EngagementTuple)

        Returns:
            encoded_act : (int)
                0..4:movements, 5..4+N:shoot_attack, 5+N..4+2N:collide_attack, 5+2N..4+3N:guard

        '''

        encoded_act = None

        if token_act.action_type == U.NOOP:
            encoded_act = 0
        elif isinstance(token_act, U.MovementTuple):
            encoded_act = U.MOVEMENT_TYPES.index(token_act.action_type)
        elif isinstance(token_act, U.EngagementTuple):

            # parse target number
            _, _, targ_num = koth.parse_token_id(token_act.target)
            targ_num = int(targ_num)

            if token_act.action_type == U.SHOOT:
                encoded_act = self.act_space_info.shoot_slice[0] + targ_num
            elif token_act.action_type == U.COLLIDE:
                encoded_act = self.act_space_info.collide_slice[0] + targ_num
            elif token_act.action_type == U.GUARD:
                encoded_act = self.act_space_info.guard_slice[0] + targ_num
            else:
                raise ValueError('Unrecognized action_type: {}'.format(token_act.action_type))

        else:
            raise ValueError('Unexpected token action type {}'.format(token_act))

        return encoded_act


class KOTHActionSpaces:
    ''' organizes the various KOTHGame observation components into gym spaces etc.)
    
    Notes:
        - Assumes and enforces that both players have equal number of tokens and identical
        action spaces
        - This class only defines the shape of the action spaces, it does not enforce the
        ordering of data values within action space samples. See the step() functions which
        is responsible for mapping gym actions into the KOTHGame input
        That said, the intented (but not enforced) ordering is given in __init__()
    '''

    def __init__(self, game: koth.KOTHGame):
        '''
        Args:
            game : koth.KOTHGame
                king-of-the-hill game object

        Attributes:
            per_token : Discrete
                action space for a single token with intended (but not enforced) ordering:
                0..4:movements, 5..4+N:shoot_attack, 5+N..4+2N:collide_attack, 5+2N..4+3N:guard
            per_player : Tuple
                action space for a single player with intended (but not enforced) ordering following
                the token number in the token_id of KOTHGame

        '''

        # enforce identical action spaces
        #if game.n_tokens_alpha != game.n_tokens_beta:
        #    raise NotImplementedError('Both players must have same number of tokens')
        self.n_tokens_per_player = game.n_tokens_alpha

        # create list of actions that maps index to discrete gym action
        # self.action_map = deepcopy(U.MOVEMENT_TYPES)

        # compute ranges of discrete actions mapping to action types
        self.move_slice = (0, len(U.MOVEMENT_TYPES))
        self.shoot_slice = (self.move_slice[1], self.move_slice[1] + self.n_tokens_per_player)
        self.collide_slice = (self.shoot_slice[1], self.shoot_slice[1] + self.n_tokens_per_player)
        self.guard_slice = (self.collide_slice[1], self.collide_slice[1] + self.n_tokens_per_player)

        # action space for an individual token
        # 0..4:movements, 5..4+N:shoot_attack, 5+N..4+2N:collide_attack, 5+2N..4+3N:guard
        n_acts_per_token = len(U.MOVEMENT_TYPES) + len(U.ENGAGEMENT_TYPES) * self.n_tokens_per_player
        self.per_token = spaces.Discrete(n_acts_per_token)
        self.flat_per_token = spaces.flatten_space(self.per_token)
        assert self.flat_per_token.shape == (N_BITS_ACT_PER_TOKEN,)

        # action space for each player
        self.per_player = spaces.Tuple(
            tuple([self.per_token for _ in range(self.n_tokens_per_player)]))

        # action mask for each player
        self.mask_per_player = spaces.Box(
            low=0, high=1,
            shape=(self.n_tokens_per_player, n_acts_per_token),
            dtype=self.flat_per_token.dtype
        )
        assert self.mask_per_player.shape == (self.n_tokens_per_player, N_BITS_ACT_PER_TOKEN)

        # NOTE: don't use flat_per_player for action space 
        # because we cannot sample valid actions from the flat space
        self.flat_per_player = spaces.flatten_space(self.per_player)
        assert self.flat_per_player.shape == (N_BITS_ACT_PER_PLAYER,)


class KOTHObservationSpaces:
    ''' organizes the various KOTHGame observation components into gym spaces etc.)'''

    def __init__(self, game: koth.KOTHGame):
        '''
        Args:
            game : koth.KOTHGame
                king-of-the-hill game object

        TODO: verify listed attributes match implementation
        Attributes:
            score : MultiBinary()
                player's cumulative score in game
            turn_count : MultiBinary()
                current turn count of game
            turn_phase : Discrete(3)
                current phase of turn (movement, engagement, drift)
            own_hill : MultiBinary(1)
                whether it's your hill position (1) or opponents(0)
            hill_position : Discrete()
                position of hill location
            hill : Tuple()
                combination of hill ownership and location
            own_piece : MultiBinary(1)
                whether it's your piece (1) or opponents (0)
            role : Discrete(2) 
                role of piece (seeker=0, bludger=1)
            position : Discrete()
                location of piece on board
            fuel : MultiBinary() 
                binary representation of fuel remaining, rounded down to nearest int
            ammo : MultiBinary()
                binary representation of ammo remaining
            per_token : Tuple()
                complete observation space of a single token
            tokens_per_player : Tuple()
                observation space of all tokens
            per_player : Dict()
                complete observation space of a single player 
                (score, own hill, own tokens, opponent hill, opponent tokens)
        '''

        if game.n_tokens_alpha != game.n_tokens_beta:
            raise NotImplementedError('Both playe must have same number of tokens')
        n_tokens_per_player = game.n_tokens_alpha

        # non-token game observations
        turn_phase = spaces.Discrete(len(U.TURN_PHASE_LIST))
        turn_count = spaces.MultiBinary(len(U.int2bitlist(game.inargs.max_turns)))
        ownership = spaces.MultiBinary(1)
        score = spaces.MultiBinary(max(
            len(U.int2bitlist(int(max(game.inargs.win_score[U.P1],game.inargs.win_score[U.P2])))),
            1 + len(U.int2bitlist(int(abs(game.inargs.illegal_action_score))))
        ))
        score = spaces.Tuple((ownership, score))
        hill = spaces.Discrete(game.board_grid.n_sectors)
        hill = spaces.Tuple((ownership, hill))
        self.scoreboard = spaces.Dict({
            'turn_phase': turn_phase,
            'turn_count': turn_count,
            'own_score': score,
            'opponent_score': score,
            'own_hill': hill,
            'opponent_hill': hill})
        self.flat_scoreboard = spaces.flatten_space(self.scoreboard)
        assert self.flat_scoreboard.shape == (N_BITS_OBS_SCOREBOARD,)

        # observation of single token
        self.per_token_components = TokenComponentSpaces(
            own_piece=spaces.MultiBinary(1),
            role=spaces.Discrete(len(U.PIECE_ROLES)),
            position=spaces.Discrete(game.board_grid.n_sectors),
            fuel=spaces.MultiBinary(max(
                len(U.int2bitlist(int(game.inargs.init_fuel[U.P1][U.SEEKER]))),
                len(U.int2bitlist(int(game.inargs.init_fuel[U.P1][U.BLUDGER]))),
                len(U.int2bitlist(int(game.inargs.init_fuel[U.P2][U.SEEKER]))),
                len(U.int2bitlist(int(game.inargs.init_fuel[U.P2][U.BLUDGER]))))),
            ammo=spaces.MultiBinary(max(
                len(U.int2bitlist(game.inargs.init_ammo[U.P1][U.SEEKER])),
                len(U.int2bitlist(game.inargs.init_ammo[U.P1][U.BLUDGER])),
                len(U.int2bitlist(game.inargs.init_ammo[U.P2][U.SEEKER])),
                len(U.int2bitlist(game.inargs.init_ammo[U.P2][U.BLUDGER]))))
        )

        # flattened observation of single token (1D vector normalized elements to range [0,1])
        self.flat_per_token_components = TokenComponentSpaces(
            own_piece=spaces.flatten_space(self.per_token_components.own_piece),
            role=spaces.flatten_space(self.per_token_components.role),
            position=spaces.flatten_space(self.per_token_components.position),
            fuel=spaces.flatten_space(self.per_token_components.fuel),
            ammo=spaces.flatten_space(self.per_token_components.ammo)
        )

        # check that flat space dimensions matach expectation
        assert self.flat_per_token_components.own_piece.shape == (N_BITS_OBS_OWN_PIECE,)
        assert self.flat_per_token_components.role.shape == (N_BITS_OBS_ROLE,)
        assert self.flat_per_token_components.position.shape == (N_BITS_OBS_POSITION,)
        assert self.flat_per_token_components.fuel.shape == (N_BITS_OBS_FUEL,)
        assert self.flat_per_token_components.ammo.shape == (N_BITS_OBS_AMMO,)

        # Complete observation space of a single token
        self.per_token = spaces.Tuple((
            self.per_token_components.own_piece,
            self.per_token_components.role,
            self.per_token_components.position,
            self.per_token_components.fuel,
            self.per_token_components.ammo
        ))
        self.flat_per_token = spaces.flatten_space(self.per_token)
        assert self.flat_per_token.shape == (N_BITS_OBS_PER_TOKEN,)

        # Observation space of all tokens
        self.tokens_per_player = spaces.Tuple(
            [self.per_token for _ in range(n_tokens_per_player)])
        self.flat_tokens_per_player = spaces.flatten_space(self.tokens_per_player)
        assert self.flat_tokens_per_player.shape == (N_BITS_OBS_TOKENS_PER_PLAYER,)

        # all observations per player
        self.per_player = spaces.Dict({
            'scoreboard': self.scoreboard,
            'own_tokens': self.tokens_per_player,
            'opponent_tokens': self.tokens_per_player,
        })
        self.flat_per_player = spaces.flatten_space(self.per_player)
        assert self.flat_per_player.shape == (N_BITS_OBS_PER_PLAYER,)


def get_non_negative_binary_observation(val, n_bits):
    '''convert non negative int or float value to fixed-length, little-endian binary array'''

    # convert value to int then tovariable-length binary list
    assert val >= 0
    val_bin = U.int2bitlist(int(val))
    n_bits_val = len(val_bin)

    if n_bits_val > n_bits:
        raise ValueError('Value {} too large to be represented with {} bits'.format(val, n_bits))

    obs_val = np.zeros(n_bits)
    obs_val[-n_bits_val:] = val_bin

    return obs_val


def get_binary_observation(val, n_bits):
    '''convert int or float value to fixed-length, little-endian binary array

    Negative numbers start with 1, positive start with zero
    '''

    # convert value to int then tovariable-length binary list
    is_neg = val < 0
    val_bin = U.int2bitlist(int(abs(val)))
    n_bits_val = len(val_bin) + 1

    if n_bits_val > n_bits:
        raise ValueError('Value {} too large to be represented with {} bits'.format(val, n_bits))

    obs_val = np.zeros(n_bits, 'int')
    obs_val[0] = is_neg
    obs_val[-len(val_bin):] = val_bin

    return obs_val
