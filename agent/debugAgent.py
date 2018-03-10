from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table, card_to_normal_str
import random

CHAR_SUIT_TO_INT = {
        's': 0,  # spades
        'h': 13,  # hearts
        'd': 26,  # diamonds
        'c': 39,  # clubs
}
# " {'23456789TJQKA'} + {'shdc''} (note: lower case) "
CHAR_NUM_TO_INT = {
        '2': 1,
        '3': 2,
        '4': 3,
        '5': 4,
        '6': 5,
        '7': 6,
        '8': 7,
        '9': 8,
        'T': 9,
        'J': 10,
        'Q': 11,
        'K': 12,
        'A': 0,
}

class debugModel():
    def __init__(self):
        self._nothing = "test"
        self.reload_left = 2
        self.model = {"seed":831}
        self._state = [0] * 52 * 2 + [0] * 52 * 5 + [0] *3 # { my 2 card (one hot), community 5 card (one hot), total_pot, my_stack, to_call) ]

    def __turn_card_to_one_hot(self, card):
        if card == -1:
            return [0] * 52
        # " {'23456789TJQKA'} + {'shdc''} (note: lower case) "
        card_info = card_to_normal_str(card)
        card_idx = CHAR_NUM_TO_INT[card_info[:1]] + CHAR_SUIT_TO_INT[card_info[1:]]
        card_hot = [0]*52
        card_hot[card_idx] = 1
        return card_hot

    def __turn_observation_to_state(self, observation, playerid):
        my_card = observation.player_states[playerid].hand
        community_card = observation.community_card
        my_stack = observation.player_states[playerid].stack
        total_pot = observation.community_state.totalpot
        to_call = observation.community_state.to_call
        return self.__turn_card_to_one_hot(my_card[0]) + \
               self.__turn_card_to_one_hot(my_card[1])+ \
               self.__turn_card_to_one_hot(community_card[0])+ \
               self.__turn_card_to_one_hot(community_card[1])+ \
               self.__turn_card_to_one_hot(community_card[2])+ \
               self.__turn_card_to_one_hot(community_card[3])+ \
               self.__turn_card_to_one_hot(community_card[4])+ \
               [total_pot, my_stack, to_call]

    def batchTrainModel(self):
        return

    def onlineTrainModel(self):
        return

    def saveModel(self, path):
        return

    def loadModel(self, path):
        return

    def takeAction(self, state, playerid):
        ''' (Predict/ Policy) Select Action under state'''
        print("debug >>>")
        for p in state.player_states:
            print(p)
        print(state.community_state)
        print(state.community_card)
        print(playerid)
        print("<<<  ")
        # print("Test State : ", self.__turn_observation_to_state(state, playerid))
        if state.community_state.to_call > 0:
            if random.random() > 0.7 :
                return ACTION(action_table.FOLD, 0)
            else:
                return ACTION(action_table.CALL, state.community_state.to_call)
        else:
            if random.random() > 0.7:
                return ACTION(action_table.RAISE, 50)
            elif random.random() > 0.9:
                return ACTION(action_table.RAISE, state.player_states[playerid].stack)
            else:
                return ACTION(action_table.CHECK, 0)

    def getReload(self, state):
        '''return `True` if reload is needed under state, otherwise `False`'''
        if self.reload_left > 0:
            self.reload_left -= 1
            return True
        else:
            return False