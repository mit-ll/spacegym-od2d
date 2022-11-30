# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# this is meant as a sandbox for running a complete koth game
# with random-yet-valid agents

import numpy as np
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
from orbit_defender2d.king_of_the_hill import koth

def run_pettingzoo_random_game():

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
        koth.print_actions(actions)

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
    print("\n====GAME FINISHED====\nWinner: {}\nScore: {}|{}\n=====================\n".format(winner, alpha_score, beta_score))

if __name__ == "__main__":
    run_pettingzoo_random_game()
