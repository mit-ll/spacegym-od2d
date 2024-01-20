# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# Default Game Parameters 
# used for default KOTHGame instantiation
# and Gym environment versioning
# Note: a change to any of these params implies a change to the gym environment
# version number

# If playing an asymmetric game, then set ASYMMETRIC_GAME to True
# In an aymmetric game, this file must have the largest values to be expected in the following parameters:
    # All board parameters (MAX_RING, MIN_RING, GEO_RING, NUM_SPACES, INIT_BOARD_PATTERN, NUM_TOKENS_PER_PLAYER)
    # Initial ammo and fuel (INIT_FUEL, INIT_AMMO)
# If using or training an AI, pettingzoo_env.py will use the values in this file to set the observation and action space sizes.
# After the game is initialized, the game will read in and set asymmetric palyer values based on another file called asymmetric_game_parameters.py

import orbit_defender2d.utils.utils as U

ASYMMETRIC_FLAG = True

# board sizing
MAX_RING = 5
MIN_RING = 1
GEO_RING = 4
if MIN_RING == 1:
    NUM_SPACES = 2**(MAX_RING + 1) -2**(MIN_RING) #Get the number of spaces in the board (not including the center)
elif MIN_RING > 1:
    NUM_SPACES = 2**(MAX_RING + 1) -2**(MIN_RING - 1) #Get the number of spaces in the board (not including the center)
else:
    raise ValueError("MIN_RING must be >= 1")

# initial token placement and attributes
INIT_BOARD_PATTERN = [(-2,1), (-1,3), (0,2), (1,3), (2,1)] # (relative azim, number of pieces)

NUM_TOKENS_PER_PLAYER = sum([a[1] for a in INIT_BOARD_PATTERN])+1 #Get the number of tokens per player, plus 1 for the seeker

INIT_FUEL = {
    U.SEEKER:   100.0,
    U.BLUDGER:  100.0,
}
INIT_AMMO = {
    U.SEEKER:   0,
    U.BLUDGER:  1,
}

# engagement and movement parameters
MIN_FUEL = 0.0
FUEL_USAGE = {
    U.NOOP: 0.0,
    U.DRIFT: 1.0, #Essentially the cost of station keeping
    U.PROGRADE: 5.0, #make this lower than radial in/out was 10
    U.RETROGRADE: 5.0,
    U.RADIAL_IN: 10.0, #These should be higher, probably 10, was 5
    U.RADIAL_OUT: 10.0,
    U.IN_SEC:{
        U.SHOOT: 5.0,
        U.COLLIDE: 20.0,
        U.GUARD: 5.0
    },
    U.ADJ_SEC:{
        U.SHOOT: 7.0, #increased from 5
        U.COLLIDE: 30.0, #Should be higher, was 20
        U.GUARD: 10.0
    }
}
ENGAGE_PROBS = {
    U.IN_SEC:{
        U.NOOP: 1.0,
        U.SHOOT: 0.7, 
        U.COLLIDE: 0.8,
        U.GUARD: 0.9},
    U.ADJ_SEC:{
        U.NOOP: 1.0,
        U.SHOOT: 0.3, #These should be lower, maybe 0.3,0.4,0.5
        U.COLLIDE: 0.4,
        U.GUARD: 0.5
    }
}

# scoring and game termination
ILLEGAL_ACT_SCORE = -1000.0
IN_GOAL_POINTS = 10.0
ADJ_GOAL_POINTS = 3.0
FUEL_POINTS_FACTOR = 1.0
FUEL_POINTS_FACTOR_BLUDGER = 0.1
WIN_SCORE = 250.0
MAX_TURNS = 50 #reduced from 100 to 50

# Derived parameters
# NOTE: don't do this! any derived params should be member variables! otherwise
# if any of these default params are overwritten in a object instance, 
# the derived params won't be updated
# _N_SECTORS_TOTAL = 2**(PARAM_MAX_RING + 1) -1
# _N_TOKENS = sum([a[1] for a in INIT_BOARD_PATTERN]) * N_PLAYERS
