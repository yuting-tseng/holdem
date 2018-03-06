from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table, card_to_normal_str
import random

from treys import Evaluator
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras import backend as K
# from keras.losses import binary_crossentropy

CHAR_SUIT_TO_INT = {
        's': 0,  # spades
        'h': 13,  # hearts
        'd': 26,  # diamonds
        'c': 39,  # clubs
}
# " {'23456789TJQKA'} + {'shdc''} (note: lower case) "
CHAR_NUM_TO_INT = {
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        'T': 10,
        'J': 11,
        'Q': 12,
        'K': 13,
        'A': 1,
}

class dqnModel():
    def __init__(self,env):
        self._nothing = "test"
        self.reload_left = 2
        self.model = {"seed":831}

        # total 367 states
        self._state = [0] * 52 * 2 + [0] * 52 * 5 + [0] *3 # { my 2 card (one hot), community 5 card (one hot), total_pot, my_stack, to_call) ]
        # add new initial
        self.action_size = 4
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        

        # self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.load_weights('breakout_pass.h5')

    def _build_model(self):
        model = Sequential()

        model.add(Dense(100, input_dim=784))
        model.add(Dense(50, input_dim=100))
        model.add(Dense(self.action_size, input_dim=50))

        opt = Adam(lr=self.learning_rate)
        #opt=RMSprop(lr=self.learning_rate,decay=0.99)
        model.compile(loss=self._huber_loss, optimizer=opt)
        model.summary()
        return model

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
        print("state",state)
        print("playerid",playerid)
        print("Test State : ", self.__turn_observation_to_state(state, playerid))
        input("pause")
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