# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from collections import namedtuple

# unified variable names
P1 = 'alpha'
P2 = 'beta'
TOKEN_DELIMITER = ':'
GOAL1 = 'goal_sector_alpha'
GOAL2 = 'goal_sector_beta'
TOKEN_STATES = 'token_states'
TURN_COUNT = 'turn_count'
TURN_PHASE = 'turn_phase'
GAME_DONE = 'game_done'
SCORE = 'score'
FUEL_SCORE = 'fuel_score'
TOKEN_ADJACENCY = 'token_adjacency_graph'
LEGAL_ACTIONS = 'legal_verbose_actions'
TURN_PHASE_LIST = [MOVEMENT, ENGAGEMENT, DRIFT] = ['movement', 'engagement', 'drift']
PIECE_ROLES = [SEEKER, BLUDGER] = ['HVA', 'Patrol']
MOVEMENT_TYPES = [NOOP, PROGRADE, RETROGRADE, RADIAL_IN, RADIAL_OUT] = \
                ['noop', 'prograde', 'retrograde', 'radial_in', 'radial_out']
ENGAGEMENT_TYPES = [SHOOT, COLLIDE, GUARD] = ['shoot', 'collide', 'guard']
[IN_SEC, ADJ_SEC] = ['in_sector', 'adjacent_sector']
INVALID_ACTION = 'invalid_action'
# TCP_PORT = 'tcp_port'
# JSON_PROPERTY_NAMES = [JSON_ENG_RES_SEQ] = ['engagementResolutionSequence']


MovementTuple = namedtuple('MovementTuple', ['action_type'])
EngagementTuple = namedtuple('EngagementTuple', ['action_type', 'target', 'prob'])
EngagementOutcomeTuple = namedtuple('EngagementOutcomeTuple', ['action_type', 'attacker', 'target', 'guardian', 'prob', 'success'])
# GuardOutcomeTuple = namedtuple('GuardOutcomeTuple', ['guardian', 'target', 'attacker', 'prob', 'success'])
# ShootOutcomeTuple = namedtuple('ShootOutcomeTuple', ['attacker', 'target', 'prob', 'success'])
# CollideOutcomeTuple = namedtuple('CollideOutcomeTuple', ['attacker', 'target', 'prob', 'success'])

def int2bitlist(n):
    '''convert integer to shortest length binary in list

    Ref:
        https://stackoverflow.com/questions/10321978/integer-to-bitfield-as-a-list
    '''
    assert n >= 0
    return [1 if digit=='1' else 0 for digit in bin(n)[2:]]

def bitlist2int(b):
    '''convert binary (represented as list of bits) to integer
    '''
    n = 0
    for i, v in enumerate(reversed(b)):
        n += v * 2**i
    return n
