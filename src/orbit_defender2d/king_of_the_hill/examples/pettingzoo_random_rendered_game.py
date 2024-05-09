# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# this is meant as a sandbox for running a complete koth game
# with random-yet-valid agents

import numpy as np
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
from orbit_defender2d.king_of_the_hill import koth
from orbit_defender2d.king_of_the_hill import default_game_parameters as DGP

if __name__ == "__main__":

    # create and reset pettingzoo env
    penv = PZE.parallel_env()
    penv.reset()

    # iterate through game with valid random actions
    while True:

        print("\n<==== Turn: {} | Phase: {} ====>".format(
            penv.kothgame.game_state[U.TURN_COUNT], 
            penv.kothgame.game_state[U.TURN_PHASE]))

        # draw random legal actions:
        actions = penv.kothgame.get_random_valid_actions()
        penv.actions = actions
        koth.print_actions(actions)

        # update rendered pygame window with latency for user comprehension
        penv.render(mode="debug")

        # encode actions into flat gym space
        encoded_actions = penv.encode_all_discrete_actions(actions=actions)

        # apply encoded actions
        observations, rewards, dones, info = penv.step(actions=encoded_actions)

        # assert zero-sum game
        assert np.isclose(rewards[U.P1], -rewards[U.P2])

        # print out salient game state
        koth.print_game_info(penv.kothgame)

        # assert rewards only from final timestep
        if any([dones[d] for d in dones.keys()]):
            assert np.isclose(rewards[U.P1], 
                penv.kothgame.game_state[U.P1][U.SCORE] - penv.kothgame.game_state[U.P2][U.SCORE])
            break
        else:
            assert np.isclose(rewards[U.P1], 0.0)

    winner = None
    alpha_score = penv.kothgame.game_state[U.P1][U.SCORE]
    beta_score = penv.kothgame.game_state[U.P2][U.SCORE]
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'
    
    if penv.kothgame.game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel <= DGP.MIN_FUEL:
        term_cond = "alpha seeker out of fuel"
    elif penv.kothgame.game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel <= DGP.MIN_FUEL:
        term_cond = "beta seeker out of fuel"
    elif alpha_score >= DGP.WIN_SCORE:
        term_cond = "alpha reached Win Score"
    elif beta_score  >= DGP.WIN_SCORE:
        term_cond = "beta reached Win Score"
    elif penv.kothgame.game_state[U.TURN_COUNT]  >= DGP.MAX_TURNS:
        term_cond = "max turns reached" 
    else:
        term_cond = "unknown"

    print(
        "\n====GAME FINISHED====\n" +
        "Winner: {} \n".format(winner) + 
        "Score: {}|{}\n".format(alpha_score, beta_score) + 
        "=====================\n")
    print("Termination condition: {}".format(term_cond))

    if penv.render_active:
        penv.draw_win(winner)


