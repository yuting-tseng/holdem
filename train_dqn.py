import gym
import holdem
import agent

def add_users(env):
    model_list = list()

    # start with 4 players
    env.add_player(0, stack=5000) # add a player to seat 0 with 1000 "chips"
#     model_list.append(agent.allFoldModel())
    model_list.append(agent.MontecarloModel())

    env.add_player(1, stack=5000) # add another player to seat 1 with 2000 "chips"
#     model_list.append(agent.allFoldModel())
    model_list.append(agent.MontecarloModel())

    env.add_player(2, stack=5000) # add another player to seat 2 with 3000 "chips"
    # model_list.append(agent.allFoldModel())
    model_list.append(agent.MontecarloModel())

    env.add_player(3, stack=5000) # add another player to seat 3 with 1000 "chips"
    model_list.append(agent.allFoldModel())

    env.add_player(4, stack=5000) # add another player to seat 3 with 1000 "chips"
    model_list.append(agent.dqnModel())

    env.add_player(5, stack=5000) # add another player to seat 3 with 1000 "chips"
    model_list.append(agent.dqnModel())
#     model_list.append(agent.allFoldModel())

    # env.add_player(6, stack=1000) # add another player to seat 3 with 1000 "chips"
    # model_list.append(agent.allCallModel())

    # env.add_player(7, stack=1000) # add another player to seat 3 with 1000 "chips"
    # model_list.append(agent.MontecarloModel())

    # env.add_player(8, stack=1000) # add another player to seat 3 with 1000 "chips"
    # model_list.append(agent.allFoldModel())

    # env.add_player(9, stack=1000) # add another player to seat 3 with 1000 "chips"
    # model_list.append(agent.allinModel())

    return model_list

playerid = 4
episodes = 30
pre_train_steps = 20 #How many steps of random actions before training begins.

def lets_play():
    env = gym.make('TexasHoldem-v2') # holdem.TexasHoldemEnv(2)

    model_list = add_users(env)
    
    agent = model_list[playerid]
    total_steps = 0

    # Iterate the game
    for e in range(episodes):
    #     lets_play(env, env.n_seats, model_list)
        # reset state in the beginning of each game
        state = env.reset()
        env.render(mode='human')
        done = False
        if env.episode_end:
            break
        stack = state.player_states[playerid].stack
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        while not done:
            # play safe actions, check when no one else has raised, call when raised.
            # print(">>> Debug Information ")
            # print("state(t)")
            # for p in cur_state.player_states:
            #     print(p)
            # print(cur_state.community_state)

            actions = holdem.model_list_action(state, n_seats=env.n_seats, model_list=model_list)
            next_state, rewards, done, _ = env.step(actions)
            action = actions[playerid][0]
            reward = 0
            if state.community_state.current_player == playerid:
                agent.remember(state, action, reward, next_state, done, playerid)
#                 print('actions: ', action)
                agent.onlineTrainModel()
                env.render(mode='human')
            state = next_state
#             print("memory len: ", len(agent.memory))

    #         print('table card: ', state.community_card)

            # print("action(t), (CALL=1, RAISE=2, FOLD=3 , CHECK=0, [action, amount])")
            # print(actions)
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}".format(e, episodes, reward))
                reward = rewards[playerid] - stack
                stack = rewards[playerid]
                agent.memory.pop()
                agent.remember(state, action, reward, next_state, done, playerid)

            # print("reward(t+1)")
            # print(rews)
            # print("<<< Debug Information ")

            # print("final state")
            # print(cur_state)

            # total_stack = sum([p.stack for p in env._seats])
            # if total_stack != 10000:
            #     return

            # train the agent with the experience of the episode
        if total_steps > pre_train_steps:
            agent.replay(10)

        total_steps += 1

        agent.saveModel()

        print("Episode End!!!")


for i in range(1000):
    print(i, " plays")
    try:
        lets_play()
    except:
        print('Error!!!!')
