from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table, card_to_normal_str
import random
import os
import json
import numpy as np

from treys import Evaluator, Card
#from treys import Card
#from evaluator import Evaluator
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.losses import binary_crossentropy
from collections import deque
from keras.callbacks import History 
import logging

logging.basicConfig(level=logging.DEBUG)

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

class dqnAction():
    FOLD = 0
    CALL = 1
    RAISE_LESS = 2
    RAISE_MORE = 3
    ALLIN = 4


def getTotalCards():
    TotalNum = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    Totalflower = ['s', 'h', 'c', 'd']
    TotalCards = list()
    for f in Totalflower:
        for n in TotalNum:
            TotalCards.append(n+f)
    res = dict()
    for i, c in enumerate(TotalCards):
        res[i] = c

    CardID = {v: k for k, v in res.items()}
    return res, CardID

class Action():
    def __init__(self, state):
        self.state = state

    def Fold(self):
        #print('FOLD')
        if self.state.community_state.to_call == 0:
            return ACTION(action_table.CHECK, 0)
        else:
            return ACTION(action_table.FOLD, 0)
    
    def AllIn(self, playerid):
        #print("All in")
        return ACTION(action_table.RAISE, self.state.player_states[playerid].stack) # all in
    
    def Call(self): 
        #print("Call")
        if self.state.community_state.to_call == 0:
            return ACTION(action_table.CHECK, 0)
        else:
            return ACTION(action_table.CALL, self.state.community_state.to_call)
    
    def Raise(self, raise_upper, min_raise, raise_amount):
        #print("Raise")
        if min([raise_upper, min_raise, raise_amount]) == raise_upper: # fold
            #print("Raise amount too high")
            return self.Fold()
        if min_raise > raise_amount: # call
            #print("minimum raise more than amount to be raised")
            return self.Call()
        else:
            return ACTION(action_table.RAISE, raise_amount)

class dqnModel():
    # https://keon.io/deep-q-learning/
    def __init__(self, memory_path='DQN_memory.txt'):
        self._nothing = "test"
        self.reload_left = 2
        self.model = {"seed":831}

        # total 367 states
        #self._state = [0] * 52 * 2 + [0] * 52 * 5 + [0] *4 # { my 2 card (one hot), community 5 card (one hot), total_pot, my_stack, to_call, win_prob) ]
        self._state = [0] * 52 * 2 + [0] * 52 * 5 + [0] * 9 + [0] * 5 # { my 2 card (one hot), community 5 card (one hot), opponent's action(10 opponents), total_pot, my_stack, to_call, my_betting, win_prob) ]
        # add new initial
        self.action_size = 5
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.ModelDir = 'Model/'
        self.ModelName = 'DQNmodel.h5'
        self.loadModel()
        self.MemoryPath = self.ModelDir + memory_path
        
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # monte carlo method
        self.simulation_number = 300
        self.TotalCards, self.CardIDmapping = getTotalCards()

        # state
        self.last_state = None
        self.last_action = None
        self.stack_init = 0

        try:
            self.update_target_model()
    
    def get_ModelPath(self):
        if not os.path.isdir(self.ModelDir):
            os.mkdir(self.ModelDir)
        return self.ModelDir + self.ModelName
        

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        loss =  K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
        return loss
    
    def remember(self, state, action, reward, next_state, done, playerid):
        state = self.__turn_observation_to_state(state, playerid)
        next_state = self.__turn_observation_to_state(next_state, playerid)
        
        state = np.array(state).reshape(1,len(self._state))
        next_state = np.array(next_state).reshape(1,len(self._state))
        #self.memory.append((state, action, reward, next_state, done))
        self.memory.append((action, reward, state, next_state, done))
        
    def act(self, state, playerid):
        state = self.__turn_observation_to_state(state, playerid)
        state = np.array(state).reshape(1,len(self._state))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        #print("DQN action: ", act_values)
        return np.argmax(act_values[0])  # returns action
    
    def _replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.load_weights(self.get_ModelPath())

    def _build_model(self):
        model = Sequential()

        model.add(Dense(1024, input_dim=len(self._state)))
#         model.add(Dense(100, input_shape=(len(self._state),)))
        model.add(Dense(512, input_dim=1024))
        model.add(Dense(256, input_dim=512))
        model.add(Dense(64, input_dim=64))
        model.add(Dense(16, input_dim=16))
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
        
        # for j in ['s','h','d','c']:
        #     for i in ['A','2','3','4','5','6','7','8','9','T','J','Q','K']:
        #         card_hot = [0]*52
        #         print('{0}'.format(i+j))
        #         card_idx = CHAR_NUM_TO_INT[i] + CHAR_SUIT_TO_INT[j]
        #         print(card_idx)
        #         card_hot[card_idx] = 1
        #         print(card_hot)
        #         input('pause')

        card_idx = CHAR_NUM_TO_INT[card_info[:1]] + CHAR_SUIT_TO_INT[card_info[1:]]
        card_hot = [0]*52
        card_hot[card_idx] = 1
        return card_hot

    def _pick_unused_card(self, used_card):
        unused = self.TotalCards.keys() - set(used_card)
        return [Card.new(self.TotalCards[card_id]) for card_id in unused]

    def get_win_prob(self, state, playerid):
        """Calculate the win probability from your board cards and hand cards by using simple Monte Carlo method.
        """
        evaluator = Evaluator()

        def get_card_class(card_int_list):
            res = [Card.new(Card.int_to_str(c)) for c in card_int_list if c != -1]
            return res

        def WinProbability(hand, board):
            rank = evaluator.evaluate(board, hand)
            percentage = 1.0 - evaluator.get_five_card_rank_percentage(rank)
            return percentage

        hand_cards = get_card_class(state.player_states[playerid].hand)
        board_cards = get_card_class(state.community_card)
        if any([True for h in hand_cards if h in board_cards]):
            Card.print_pretty_cards(hand_cards)
            Card.print_pretty_cards(board_cards)
        num_players = len([ p for p in state.player_states if not p.emptyplayer])

        win = 0
        round = 0

        board_cards_to_draw = 5 - len(board_cards)  # 2
        rest_cards = self._pick_unused_card(board_cards + hand_cards)
        #print("rest cards")
        #Card.print_pretty_cards(rest_cards)
        
        #choiced = random.sample(unused, card_num)
        
        for i in range(self.simulation_number):

            unused_cards = random.sample(rest_cards, (num_players - 1) * 2 + board_cards_to_draw)
            board_sample = unused_cards[len(unused_cards)-board_cards_to_draw:]
            unused_cards = unused_cards[:len(unused_cards)-board_cards_to_draw]

            opponents_hole = [unused_cards[2 * i:2 * i + 2] for i in range(num_players - 1)]

            try:
                opponents_score = [WinProbability(hole, board_sample) for hole in opponents_hole]
                my_rank = WinProbability(hand_cards, board_sample)
                if my_rank >= max(opponents_score):
                    win += 1
                round+=1
            except Exception as inst:# Exception, e:
                #print e.message
                continue
        #print("Win:{}".format(win))
        #print('round:{}'.format(round))
        if round == 0: 
            if len(board_cards) > 1:
                try:
                    return WinProbability(board_cards, hand_cards)
                except:
                    return 0.6
            else: 
                return 0.6
        win_prob = win / float(round)
        return win_prob

    def eval_card_rank(self, state, playerid):
        evaluator = Evaluator()
        
        def get_card_class(card_int_list):
            res = [Card.new(Card.int_to_str(c)) for c in card_int_list if c != -1]
            return res

        hand_cards = get_card_class(state.player_states[playerid].hand)
        board_cards = get_card_class(state.community_card)
        if len(board_cards) < 3:
            return self.get_win_prob(state, playerid)
        #Card.print_pretty_cards(board_cards + hand_cards)
        rank = evaluator.evaluate(hand_cards, board_cards)
        percentage = 1.0 - evaluator.get_five_card_rank_percentage(rank)
        #rank_class = evaluator.get_rank_class(rank)
        #class_string = evaluator.class_to_string(rank_class)
        #percentage = 1.0 - evaluator.get_five_card_rank_percentage(rank)  # higher better here
        #return rank, percentage
        return percentage

    def get_opponent_action(self, state, playerid):
        actions = list()
        for p in state.player_states:
            if p.seat == playerid:
                continue
            if p.last_action == 'call':
                actions.append(dqnAction.CALL)
            elif p.last_action == 'check':
                actions.append(dqnAction.CALL)
            elif p.last_action == 'raise':
                if p.betting / (p.stack+0.001) > 0.5: # prevent p.stack == 0 error
                    actions.append(dqnAction.RAISE_MORE)
                else:
                    actions.append(dqnAction.RAISE_LESS)
            elif p.last_action == 'fold':
                actions.append(dqnAction.FOLD)
            else:
                actions.append(4)
        return actions

    def __turn_observation_to_state(self, observation, playerid):
        my_card = observation.player_states[playerid].hand
        community_card = observation.community_card
        my_stack = observation.player_states[playerid].stack
        my_betting = observation.player_states[playerid].betting
        total_pot = observation.community_state.totalpot
        to_call = observation.community_state.to_call
        win_rate = self.eval_card_rank(observation, playerid)
        #rank = self.eval_card_rank(observation, playerid)
        #win_rate = self.get_win_prob(observation, playerid)
        opponent_action = self.get_opponent_action(observation, playerid)
        return self.__turn_card_to_one_hot(my_card[0]) + \
               self.__turn_card_to_one_hot(my_card[1])+ \
               self.__turn_card_to_one_hot(community_card[0])+ \
               self.__turn_card_to_one_hot(community_card[1])+ \
               self.__turn_card_to_one_hot(community_card[2])+ \
               self.__turn_card_to_one_hot(community_card[3])+ \
               self.__turn_card_to_one_hot(community_card[4])+ \
               opponent_action+ \
               [total_pot, my_stack, to_call, my_betting, win_rate]

    def __turn_observation_to_stateJust52(self, observation, playerid):
        card_hot = [0]*52
        my_card = observation.player_states[playerid].hand
        for i in my_card:
            card_hot = self.__turn_card_to_one_hot_returnIndx(i, card_hot)
        community_card = observation.community_card
        for i in community_card:
            card_hot = self.__turn_card_to_one_hot_returnIndx(i, card_hot)
        my_stack = observation.player_states[playerid].stack
        total_pot = observation.community_state.totalpot
        to_call = observation.community_state.to_call
        return card_hot + [total_pot, my_stack, to_call]

    def __turn_card_to_one_hot_returnIndx(self, card, card_hot):
        if card == -1:
            return card_hot
        else:
            card_info = card_to_normal_str(card)
            card_idx = CHAR_NUM_TO_INT[card_info[:1]] + CHAR_SUIT_TO_INT[card_info[1:]]
            if card_hot[card_idx] == 1:
                input("Error!!!!!! card_hot cann't duplicate")
            else:
                card_hot[card_idx] = 1
            return card_hot

    def batchTrainModel(self, batch_size):
        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]
                
        def get_memory():
            with open(self.MemoryPath) as f:
                content = f.readlines()
            content = [json.loads(x.strip()) for x in content] 
            return content
        
        history = History()
        memory = get_memory()
        
        for minibatch in batch(memory, batch_size):

            for action, reward, state, next_state, done in minibatch:
                state = np.array(state)
                next_state = np.array(next_state)

                target = self.model.predict(state)
                target_val = self.model.predict(next_state)
                target_val_ = self.target_model.predict(next_state)

                if done:
                    target[0][action] = reward
                else:
                    a = np.argmax(target_val)
                    target[0][action] = reward + self.gamma * target_val_[0][a]

                self.model.fit(state, target, epochs=1, verbose=0, callbacks=[history])
                print(history.history)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            self.saveModel()
        return None

    def onlineTrainModel(self):
        history = History()
        #state, action, reward, next_state, done = self.memory[-1]
        action, reward, state, next_state, done = self.memory[-1]

        target = self.model.predict(state)
        target_val = self.model.predict(next_state)
        target_val_ = self.target_model.predict(next_state)

        if done:
            target[0][action] = reward
        else:
            a = np.argmax(target_val)
            target[0][action] = reward + self.gamma * target_val_[0][a]

        self.model.fit(state, target, epochs=1, verbose=0, callbacks=[history])
        print(history.history)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        #if len(self.memory) % 20 == 0:
        self.saveModel()

    def saveMemory(self):
        memoryfile=self.MemoryPath
        with open(memoryfile, 'a') as the_file:
            while len(self.memory) > 0:
            #for line in self.memory:
                line = self.memory.popleft()
                elem = list(line)
                #elem[0] = elem[0].tolist()
                elem[2] = elem[2].tolist()
                elem[3] = elem[3].tolist()
                try:
                    the_file.write(json.dumps(elem) + '\n')
                except:
                    #print('elem: ', elem)
                    elem[0] = elem[0] * 1.0
                    elem[1] = elem[1] * 1.0
                    elem[4] = elem[4] * 1.0
                    the_file.write(json.dumps(elem) + '\n')

    def saveModel(self):
        self.model.save_weights(self.get_ModelPath())

    def loadModel(self):
        if os.path.isfile(self.get_ModelPath()):
            self.model.load_weights(self.get_ModelPath())
            #self.target_model.load_weights(self.get_ModelPath())


    def getReward(self, state, playerid):
        def get_card_class(card_int_list):
            res = [Card.new(Card.int_to_str(c)) for c in card_int_list if c != -1]
            return res

        def get_final_ranking():
            evaluator = Evaluator()
            final_ranking = list()

            for p in state.player_states:
                hand_cards = get_card_class(p.hand)
                board_cards = get_card_class(state.community_card)
                if not hand_cards: # player not play this round
                    continue

                rank = evaluator.evaluate(hand_cards, board_cards)
                final_ranking.append(rank)
            return final_ranking

        action = Action(state)
        final_ranking = get_final_ranking()

        reward = state.player_states[playerid].stack - self.stack_init
        # if min(final_ranking) == final_ranking[playerid] and reward <= 0: # if player could win but not win 
        #     reward = -1.0 * state.community_state.totalpot
        # elif reward > 0:
        #     if final_ranking[playerid] < 300:
        #         reward *= 15
        #     elif final_ranking[playerid] < 700:
        #         reward *= 10
        #     elif final_ranking[playerid] < 1500:
        #         reward *= 5
        #     elif final_ranking[playerid] < 3000:
        #         reward *= 3
        return reward

    def RoundEndAction(self, state, playerid): 
        reward = self.getReward(state, playerid)
        done = 1
        self.remember(self.last_state, self.last_action, reward, state, done, playerid)
        self.onlineTrainModel()
        self.saveMemory()

        self.last_state = None
        self.last_action = None
        return None

    def takeAction(self, state, playerid):
        ''' (Predict/ Policy) Select Action under state'''

        # load Model parameters
        self.loadModel()
        
        action = Action(state)

        #print('last state none :', self.last_state == None)
        if self.last_state == None:
            win_rate = self.get_win_prob(state, playerid)
            call_upper = win_rate * state.player_states[playerid].stack * 0.3
            self.last_state = state
            self.stack_init = state.player_states[playerid].stack + state.player_states[playerid].betting
            if call_upper > state.community_state.to_call:
                self.last_action = dqnAction.CALL
                return action.Call()
            else:
                self.last_action = dqnAction.FOLD
                return action.Fold()

        reward = 0
        self.remember(self.last_state, self.last_action, reward, state, 0, playerid)
        self.onlineTrainModel()

        self.last_state = state
        
        min_raise = max(state.community_state.lastraise * 2, state.community_state.bigblind) 
        raise_upper = state.player_states[playerid].stack / 4

        stack = state.player_states[playerid].stack
        if stack > 5000:
            stack = stack % 3000 

        react = self.act(state, playerid)
        self.last_action = react
        #print('DQN action: ', react)
        if react == dqnAction.FOLD:
            return action.Fold()
        elif react == dqnAction.CALL:# and state.community_state.to_call < int(stack / 15):
            return action.Call()
        elif react == dqnAction.RAISE_LESS:
            raise_amount = state.community_state.to_call + int(stack / 25)
            return action.Raise(raise_upper, min_raise, raise_amount)
        elif react == dqnAction.RAISE_MORE:
            raise_amount = state.community_state.to_call + int(stack / 10)
            return action.Raise(raise_upper, min_raise, raise_amount)
        elif react == dqnAction.ALLIN:
            return action.AllIn(playerid)
        else: 
            raise ValueError('react not found')

    def getReload(self, state):
        '''return `True` if reload is needed under state, otherwise `False`'''
        if self.reload_left > 0:
            self.reload_left -= 1
            return True
        else:
            return False

    def sameSuit(self, _stateCards):
        x = np.array(_stateCards[:53])
        print(x)
        print(np.where(x == 1))
        
        input("pause")


