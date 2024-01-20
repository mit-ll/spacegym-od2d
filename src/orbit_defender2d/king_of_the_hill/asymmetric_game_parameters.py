# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# Asymmetric Game Parameters 
# used for creating differences between player 1 and player 2 in KOTHGame instantiation

# If playing an asymmetric game, then set ASYMMETRIC_GAME to True
# In an aymmetric game, this file must have the largest values to be expected in the following parameters:
    # All board parameters (MAX_RING, MIN_RING, GEO_RING, NUM_SPACES, INIT_BOARD_PATTERN, NUM_TOKENS_PER_PLAYER)
    # Initial ammo and fuel (INIT_FUEL, INIT_AMMO)
# If using or training an AI, pettingzoo_env.py will use the values in this file to set the observation and action space sizes.
# After the game is initialized, the game will read in and set asymmetric palyer values based on another file called asymmetric_game_parameters.py

#TODO: Reamining to implement in KOTH.py, asymmetric fuel usage. Fuel usge may not be important since we can set the initial fuel. 
#TODO: Get rid of the need for the asymmetric_game_parameters.py file and instead put all of this info into the default_game_parameters.py file and adjust inargs to simply include all the different player parameters.

import orbit_defender2d.utils.utils as U

################### Player 1 parameters ###################
INIT_BOARD_PATTERN_P1 = [(-2,1), (-1,3), (0,2), (1,3), (2,1)] # (relative azim, number of pieces)
NUM_TOKENS_P1 = sum([a[1] for a in INIT_BOARD_PATTERN_P1])+1 #Get the number of tokens per player, plus 1 for the seeker
INIT_FUEL_P1 = {
    U.SEEKER:   100.0,
    U.BLUDGER:  100.0,
}
INIT_AMMO_P1 = {
    U.SEEKER:   0,
    U.BLUDGER:  1,
}
# engagement and movement parameters
FUEL_USAGE_P1 = {
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
ENGAGE_PROBS_P1 = {
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
IN_GOAL_POINTS_P1 = 10.0
ADJ_GOAL_POINTS_P1 = 3.0
FUEL_POINTS_FACTOR_P1 = 1.0
FUEL_POINTS_FACTOR_BLUDGER_P1 = 0.1



################### Player 2 parameters ###################
INIT_BOARD_PATTERN_P2 = [(-20,0), (-1,0), (0,2), (1,3), (2,0)] # (relative azim, number of pieces)
NUM_TOKENS_P2 = sum([a[1] for a in INIT_BOARD_PATTERN_P2])+1 #Get the number of tokens per player, plus 1 for the seeker
INIT_FUEL_P2 = {
    U.SEEKER:   100.0,
    U.BLUDGER:  100.0,
}
INIT_AMMO_P2 = {
    U.SEEKER:   0,
    U.BLUDGER:  1,
}
# engagement and movement parameters
FUEL_USAGE_P2 = {
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
ENGAGE_PROBS_P2 = {
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
IN_GOAL_POINTS_P2 = 10.0
ADJ_GOAL_POINTS_P2 = 3.0
FUEL_POINTS_FACTOR_P2 = 1.0
FUEL_POINTS_FACTOR_BLUDGER_P2 = 0.2
