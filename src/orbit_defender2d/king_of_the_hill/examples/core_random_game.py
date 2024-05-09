# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# this is meant as a sandbox for running a complete koth game
# with random-yet-valid agents

import numpy as np
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.default_game_parameters as DGP
from orbit_defender2d.king_of_the_hill import koth
from orbit_defender2d.king_of_the_hill.koth import KOTHGame

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

def print_game_info(game):
    # print("alpha player state: ")
    # for tok in game.game_state[U.P1][U.TOKEN_STATES]:
    #     print("-->{} | fuel: {} | position: {}".format(tok.satellite.fuel, tok.position))
    print("STATES:")
    for toknm, tok in game.token_catalog.items():
        print("   {:<16s}| position: {:<4d}| fuel: {:<8.1f} ".format(toknm, tok.position, tok.satellite.fuel))
    print("alpha|beta score: {}|{}".format(game.game_state[U.P1][U.SCORE],game.game_state[U.P2][U.SCORE]))

def print_actions(actions):
    print("ACTIONS:")
    if actions is None:
        print("   None")
    else:
        for toknm, act in actions.items():
            print("   {:<15s} | {}".format(toknm, act))


def run_core_random_game():

    # create and initialize game
    game = KOTHGame(**GAME_PARAMS._asdict())
    # game.game_state, game.token_catalog, game.n_tokens_alpha, game.n_tokens_beta = \
    #     game.initial_game_state(init_pattern_alpha=INIT_BOARD_PATTERN_2, init_pattern_beta=INIT_BOARD_PATTERN_2)

    # iterate through game with valid random actions
    while not game.game_state[U.GAME_DONE]:

        print("\n<==== Turn: {} | Phase: {} ====>".format(game.game_state[U.TURN_COUNT], game.game_state[U.TURN_PHASE]))

        # draw random legal actions:
        actions = game.get_random_valid_actions()
        # actions = None
        # if game.game_state[U.TURN_PHASE] != U.DRIFT:
        #     actions = {t:a[np.random.choice(len(a))] for t, a in game.game_state[U.LEGAL_ACTIONS].items()}

        #     # apply appropriate probabilities for engagements
        #     if game.game_state[U.TURN_PHASE] == U.ENGAGEMENT:
        #         actions = {t:U.EngagementTuple(
        #             action_type=a.action_type, 
        #             target=a.target, 
        #             prob=game.get_engagement_probability(t, a.target, a.action_type)) for  t, a in actions.items()}

        print_actions(actions)

        # apply actions
        game.apply_verbose_actions(actions=actions)

        # print out salient game state
        print_game_info(game)

    winner = None
    alpha_score = game.game_state[U.P1][U.SCORE]
    beta_score = game.game_state[U.P2][U.SCORE]
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'
    print("\n====GAME FINISHED====\nWinner: {}\nScore: {}|{}\n=====================\n".format(winner, alpha_score, beta_score))

if __name__ == "__main__":
    run_core_random_game()
