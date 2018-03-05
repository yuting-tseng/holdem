# -*- coding: utf-8 -*-
from gym.envs.registration import register

from .env import TexasHoldemEnv
from .utils import card_to_str, hand_to_str, safe_actions, model_list_action, action_table, card_str_to_list, PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION
from .TMutils import ClientPlayer

register(
	id='TexasHoldem-v0',
	entry_point='holdem.env:TexasHoldemEnv',
	kwargs={'n_seats': 2, 'debug': False},
)

register(
	id='TexasHoldem-v1',
	entry_point='holdem.env:TexasHoldemEnv',
	kwargs={'n_seats': 4, 'debug': False},
)

register(
	id='TexasHoldem-v2',
	entry_point='holdem.env:TexasHoldemEnv',
	kwargs={'n_seats': 10, 'debug': False},
)
