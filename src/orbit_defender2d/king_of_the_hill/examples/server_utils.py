# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.game_server as GS

def assert_valid_game_state(game_state):
    '''check response from game server gives valid game state
    '''
    
    # turn counter
    assert isinstance(game_state[GS.TURN_NUMBER], int)
    assert game_state[GS.TURN_NUMBER] >= 0

    # turn phase
    assert game_state[GS.TURN_PHASE] in U.TURN_PHASE_LIST

    # game done
    assert isinstance(game_state[GS.GAME_DONE], bool)

    # goal positions
    assert isinstance(game_state[GS.GOAL_ALPHA], int)
    # assert 0 < game_state[GS.GOAL_ALPHA] <= game_board.n_sectors
    assert isinstance(game_state[GS.GOAL_BETA], int)
    # assert 0 < game_state[GS.GOAL_BETA] <= game_board.n_sectors

def print_game_info(game_state):
    '''
    Print the game state information from the game server.
    This is the game server version of game state, so not compatible with the kothgame game state.
    '''
    print("STATES:")
    for tok in game_state[GS.TOKEN_STATES]:
        print("   {:<16s}| position: {:<4d}| fuel: {:<8.1f} ".format(tok[GS.PIECE_ID], tok[GS.POSITION], tok[GS.FUEL]))
    print("alpha|beta score: {}|{}".format(game_state[GS.SCORE_ALPHA],game_state[GS.SCORE_BETA]))

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
            success_status = "Success" if egout[GS.SUCCESS] else "Failure"
            if egout[GS.ACTION_TYPE] == U.SHOOT or egout[GS.ACTION_TYPE] == U.COLLIDE:
                print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format(
                    egout[GS.ACTION_TYPE], egout[GS.ATTACKER_ID], "", egout[GS.TARGET_ID], success_status))
            elif egout[GS.ACTION_TYPE] == U.GUARD:
                if isinstance(egout[GS.ATTACKER_ID], str):
                    print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format(
                        egout[GS.ACTION_TYPE], egout[GS.ATTACKER_ID], egout[GS.GUARDIAN_ID], egout[GS.TARGET_ID], success_status))
                else:
                    print("   {:<10s} | {:<16s} | {:<16s} | {:<16s} |---> {}".format(
                        egout[GS.ACTION_TYPE], "", egout[GS.GUARDIAN_ID], egout[GS.TARGET_ID], success_status))
            elif egout[GS.ACTION_TYPE] == U.NOOP:
                print("NOOP")
            else:
                raise ValueError("Unrecognized action type {}".format(egout[GS.ACTION_TYPE]))

def print_endgame_statsus(cur_game_state):
    '''
    Print the endgame scores, winner, and termination condition.
    '''

    winner = None
    #alpha_score = penv.kothgame.game_state[U.P1][U.SCORE]
    #beta_score = penv.kothgame.game_state[U.P2][U.SCORE]
    alpha_score = cur_game_state[GS.SCORE_ALPHA]
    beta_score = cur_game_state[GS.SCORE_BETA]
    
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'
    print("\n====GAME FINISHED====\nWinner: {}\nScore: {}|{}\n=====================\n".format(winner, alpha_score, beta_score))

    if cur_game_state[GS.TOKEN_STATES][0]['fuel'] <= DGP.MIN_FUEL:
        term_cond = "alpha out of fuel"
    elif cur_game_state[GS.TOKEN_STATES][1]['fuel'] <= DGP.MIN_FUEL:
        term_cond = "beta out of fuel"
    elif cur_game_state[GS.SCORE_ALPHA] >= DGP.WIN_SCORE[U.P1]:
        term_cond = "alpha reached Win Score"
    elif cur_game_state[GS.SCORE_BETA]  >= DGP.WIN_SCORE[U.P2]:
        term_cond = "beta reached Win Score"
    elif cur_game_state[GS.TURN_NUMBER]  >= DGP.MAX_TURNS:
        term_cond = "max turns reached" 
    else:
        term_cond = "unknown"
    print("Termination condition: {}".format(term_cond))


