# Demonstrate a CLI based player interaface, playing against a trained AI agent

import numpy as np
import orbit_defender2d.utils.utils as U
from CLI_example import CLI_example_GP as GP
from orbit_defender2d.king_of_the_hill import koth
import torch
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
from time import sleep
import concurrent.futures



GAME_PARAMS = koth.KOTHGameInputArgs(
    max_ring=GP.MAX_RING,
    min_ring=GP.MIN_RING,
    geo_ring=GP.GEO_RING,
    init_board_pattern_p1=GP.INIT_BOARD_PATTERN_P1,
    init_board_pattern_p2=GP.INIT_BOARD_PATTERN_P2,
    init_fuel=GP.INIT_FUEL,
    init_ammo=GP.INIT_AMMO,
    min_fuel=GP.MIN_FUEL,
    fuel_usage=GP.FUEL_USAGE,
    engage_probs=GP.ENGAGE_PROBS,
    illegal_action_score=GP.ILLEGAL_ACT_SCORE,
    in_goal_points=GP.IN_GOAL_POINTS,
    adj_goal_points=GP.ADJ_GOAL_POINTS,
    fuel_points_factor=GP.FUEL_POINTS_FACTOR,
    win_score=GP.WIN_SCORE,
    max_turns=GP.MAX_TURNS,
    fuel_points_factor_bludger=GP.FUEL_POINTS_FACTOR_BLUDGER,
    )


def get_engagement_dict_from_list(engagement_list):
    """
    Turns a list of engagement tuples or engagement outcome tuples into a list of dicts with the key as the token name and the tuple as the value
    """
    engagement_dict = {}
    for eng in engagement_list:
        engagement_dict[eng.attacker] = eng
    return engagement_dict

def run_game(model_path_beta):

    # create and reset pettingzoo env
    penv = PZE.parallel_env(game_params=GAME_PARAMS, training_randomize=False)
    obs = penv.reset()
    # Update rendered pygame window
    #penv.render(mode="debug")
    #Get the user's name:
    alias_valid = False
    while not alias_valid:
        alias = input("Enter player name: ")
        #Make sure that alias is a string <10 characters long and is not empty
        if isinstance(alias, str) and len(alias) <= 20 and len(alias) > 0:
            alias_valid = True
        else:
            print("Invalid alias. Alias must be a string <20 characters long and not empty.")
    print("Player {} is: {}".format(U.P1, alias))
    sleep(1)
    #start logfile
    logfile = koth.start_log_file('./CLI_example/game_log', p1_alias=alias, p2_alias="AI")

    # Render pygame window
    penv.render(mode="human")
    #penv.screen_shot(file_name="./od2d_screen_shot_new.png")
    
    model_beta = torch.jit.load(model_path_beta)
    model_beta.eval()

    # iterate through game with valid random actions
    while True:
        # Update rendered pygame window
        penv.render(mode="human")

        print("\n<==== Turn: {} | Phase: {} ====>".format(
            penv.kothgame.game_state[U.TURN_COUNT], 
            penv.kothgame.game_state[U.TURN_PHASE]))
        koth.print_scores(penv.kothgame)

        #Get actions from loaded policy to compare with actions from ray policy
        new_obs_tensor_beta = torch.tensor(obs[U.P2]['observation'], dtype=torch.float32)

        #Get Action masks
        new_obs_tensor_am_beta = torch.tensor(obs[U.P2]['action_mask'], dtype=torch.float32)
        new_obs_tensor_am_beta = new_obs_tensor_am_beta.reshape(-1,)
        
        #Concatenate the action mask and observation tensors
        new_obs_dict_beta = {'obs':torch.cat((new_obs_tensor_am_beta,new_obs_tensor_beta),dim=0).unsqueeze(0)}

        #Get the actions from the loaded policy
        acts_beta = model_beta(new_obs_dict_beta, [torch.tensor([0.0], dtype=torch.float32)], torch.tensor([0], dtype=torch.int64))      
    
        #Format the acts_model as a gym spaces touple which is the same as the action space tuple defined in 
        # pettingzoo_env.py as self.per_player = spaces.Tuple(tuple([self.per_token for _ in range(self.n_tokens_per_player)])) 
        # where self.per_token = spaces.Discrete(38) and self.n_tokens_per_player = 11
        #TODO: Get rid of hard coded values 11 and 38. 
        acts_beta_reshaped = acts_beta[0].reshape(11,38)
        acts_beta_split = torch.split(acts_beta_reshaped, 1, dim=0)
       

        acts_beta_list = []

        for i in range(11):

            acts_beta_list.append((torch.argmax(acts_beta_split[i]).item()))

        acts_beta_tuple = tuple(acts_beta_list)

        #Decode the actions from the model into the action dicts that can be used by koth
        actions_beta_dict = penv.decode_discrete_player_action(U.P2,acts_beta_tuple)

        #Get the actions from the player
        acts_received = False
        with concurrent.futures.ThreadPoolExecutor() as executor:
            t = executor.submit(koth.KOTHGame.get_input_actions, penv.kothgame, U.P1)
            while not acts_received:
                if t.done():
                    acts_received = True
                    actions_alpha_dict = t.result()
                sleep(1)
                penv.render(mode="human")

        actions = {}
        actions.update(actions_alpha_dict)
        actions.update(actions_beta_dict)
        penv.actions = actions #Add actions to the penv sot that they can be rendered
        
        #koth.print_actions(actions)
        koth.log_game_to_file(penv.kothgame, logfile=logfile, actions=actions)

        # Update rendered pygame window
        penv.render(mode="human")

        # encode actions into flat gym space
        encoded_actions = penv.encode_all_discrete_actions(actions=actions)

        # apply encoded actions
        obs, rewards, dones, info = penv.step(actions=encoded_actions)

        # assert zero-sum game
        assert np.isclose(rewards[U.P1], -rewards[U.P2])

        #If game_sate is "MOVEMENT" Then print the engagement outcomes from the prior ENGAGEMENT phase
        if penv.kothgame.game_state[U.TURN_PHASE] == U.MOVEMENT and penv.kothgame.game_state[U.TURN_COUNT] > 0:
            koth.print_engagement_outcomes(penv.kothgame.engagement_outcomes)
            engagement_outcomes_dict = get_engagement_dict_from_list(penv.kothgame.engagement_outcomes)
            penv.actions = engagement_outcomes_dict #Add actions to the penv sot that they can be rendered
            penv._eg_outcomes_phase = True
            penv.kothgame.game_state[U.TURN_PHASE] = U.ENGAGEMENT
            penv.render(mode="human")
            sleep(5)
            penv._eg_outcomes_phase = False
            penv.kothgame.game_state[U.TURN_PHASE] = U.MOVEMENT

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

    #Print final engagement outcomes
    koth.print_engagement_outcomes(penv.kothgame.engagement_outcomes)
    koth.log_game_to_file(penv.kothgame, logfile=logfile, actions=actions)

    cur_game_state = penv.kothgame.game_state
    if cur_game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel <= GP.MIN_FUEL:
        print(U.P1+" seeker out of fuel")
    if cur_game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel <= GP.MIN_FUEL:
        print(U.P2+" seeker out of fuel")
    if cur_game_state[U.P1][U.SCORE] >= GP.WIN_SCORE[U.P1]:
        print(U.P1+" reached Win Score")
    if cur_game_state[U.P2][U.SCORE]  >= GP.WIN_SCORE[U.P2]:
        print(U.P2+" reached Win Score")
    if cur_game_state[U.TURN_COUNT]  >= GP.MAX_TURNS:
        print("max turns reached")
        
    print("\n====GAME FINISHED====\nWinner: {}\nScore: {}|{}\n=====================\n".format(winner, alpha_score, beta_score))

    penv.draw_win(winner)
    
    #Ask user to press spacebar to end the game and close the pygame window
    select_valid = 0
    while not select_valid:
        selection = input("Press spacebar and then return to end game: ")
        if selection == " ":
            select_valid = 1
        else:
            print("Invalid selection. Please press spacebar")
    return




if __name__ == "__main__" or __name__ == "checkpoint_just_torch_CLI":

    model_path_beta = "./CLI_example/model_3800_smallBoard_15March.pt" #3800 iterations of board without outer ring. Trained on randagm init game params.

    run_game(model_path_beta)
