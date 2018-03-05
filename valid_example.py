# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Sam Wenke (samwenke@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import gym
import holdem
import agent

def lets_play(env, n_seats, model_list):
  cur_state = env.reset()

  # display the table, cards and all
  env.render(mode='human')

  end_of_game = False
  while not end_of_game:
    cycle_terminal = False
    while not cycle_terminal:
      #  play safe actions, check when no one else has raised, call when raised.
      # actions = holdem.safe_actions(cur_state, n_seats=n_seats)
      print("state(t)")
      for p in cur_state.player_states:
        print(p)
      print(cur_state.community_state)

      actions = holdem.model_list_action(cur_state, n_seats=n_seats, model_list=model_list)
      cur_state, rews, cycle_terminal, info = env.step(actions)

      print("action(t), (CALL=1, RAISE=2, FOLD=3 , CHECK=0, [action, amount])")
      print(actions)

      print("reward(t+1)")
      print(rews)

      env.render(mode='human')
    print("final state")
    print(cur_state)
    break

env = gym.make('TexasHoldem-v1') # holdem.TexasHoldemEnv(2)

model_list = list()

# start with 4 players
env.add_player(0, stack=1000) # add a player to seat 0 with 1000 "chips"
model_list.append(agent.idiotModel())

env.add_player(1, stack=2000) # add another player to seat 1 with 2000 "chips"
model_list.append(agent.idiotModel())

env.add_player(2, stack=3000) # add another player to seat 2 with 3000 "chips"
model_list.append(agent.idiotModel())

env.add_player(3, stack=1000) # add another player to seat 3 with 1000 "chips"
model_list.append(agent.idiotModel())


# play out a hand
lets_play(env, env.n_seats, model_list)