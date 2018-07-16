from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table
from treys import Card, Deck, Evaluator
import math
import random
import json
import numpy as np
from websocket import create_connection

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
    

class RuleBasedModel():
    def __init__(self):
        self._nothing = "test"
        self.reload_left = 2
        self.model = {"seed":831}
        self.simulation_number=100
        self.all_cards = [str(i) +j for i in list(range(2,10)) + ['T','J','Q','K','A'] for j in ['h','s','d','c']]

    def getCard(self,cardnume_code):
    	if cardnume_code == 'T':
        	card_num = 10
    	elif cardnume_code == 'J':
        	card_num = 11
    	elif cardnume_code == 'Q':
        	card_num = 12
    	elif cardnume_code == 'K':
        	card_num = 13
    	elif cardnume_code == 'A':
        	card_num = 14
    	else:
        	card_num = int(cardnume_code)
    	return card_num


    def batchTrainModel(self):
        return

    def onlineTrainModel(self):
        return

    def saveModel(self, path):
        return

    def loadModel(self, path):
        return

    def get_card_class(self,card_int_list):
        res = [Card.int_to_str(c) for c in card_int_list if c != -1]
        return res

    def _pick_unused_card(self,card_num, used_card):
        unused = set(self.all_cards) - set(used_card)
        choiced = random.sample(unused, card_num)
        return choiced


    def get_win_prob(self,state, playerid,hand_cards, board_cards,num_players):
        win = 0
        rounds=0
        evaluator = Evaluator()
        for i in range(self.simulation_number):

            board_cards_to_draw = 5 - len(board_cards)  # 2
            board_sample = board_cards + self._pick_unused_card(board_cards_to_draw,hand_cards+board_cards)
            unused_cards = self._pick_unused_card((num_players - 1) * 2, hand_cards + board_sample)
            board_sample = [Card.new(i) for i in board_sample]
            unused_cards = [Card.new(i) for i in unused_cards]
            opponents_hole = [unused_cards[2 * i:2 * i + 2] for i in range(num_players - 1)]
            #hand_sample = self._pick_unused_card(2, board_sample + hand_cards)
            try:
                opponents_score = [1 - evaluator.evaluate(hole, board_sample)/7462 for hole in opponents_hole]
                myhand_cards = [Card.new(i) for i in hand_cards]
                my_rank = 1 - evaluator.evaluate(myhand_cards, board_sample)/7462
                if my_rank >= max(opponents_score):
                    win += 1
                #rival_rank = evaluator.evaluate_hand(hand_sample, board_sample)
                rounds+=1
            except Exception as e:
                #print e.message
                continue
        win_prob = win / rounds
        return win_prob

    def first_round(self, state,hand_cards):
        hand_type = [i[1] for i in hand_cards]
        hand_num = [self.getCard(i[0]) for i in hand_cards]
        if len(set(hand_type)) == 1:
            win_prob = 0.6
        elif hand_num[0] - hand_num[1] <= 5:
            win_prob = 0.6
        elif hand_num[0] >= 8 or hand_num[1] >= 8:
            win_prob = 0.6
        else:
            win_prob = 0
        print(win_prob)
        return win_prob
    def second_round(self,state,hand_cards, board_cards):
        win_prob = 0
        hand_board = hand_cards + board_cards
        second_type = [i[1].lower() for i in hand_board]
        all_type = ['h','d','s','c']
        for i in all_type:
            if second_type.count(i) == 4:
                win_prob = 0.6
                return win_prob
        second_num = sorted([self.getCard(i[0]) for i in hand_board])
        set_num = set(list(range(second_num[0],second_num[0]+5)))
        set_same_num = set_num&set(second_num)
        if len(set_same_num) == 4:
            win_prob = 0.6
        return win_prob

    def takeAction(self, state, playerid):
        hand_cards = self.get_card_class(state.player_states[playerid].hand)
        #print("hand_cards:",hand_cards)
        board_cards = self.get_card_class(state.community_card)
        #print("board_cards",board_cards)
        player_num = len([p for p in state.player_states if not p.emptyplayer])# state.player_num
        if len(board_cards) == 0:
            win_rate = self.first_round(state,hand_cards)
            print("win Rate: ", win_rate)
        elif len(board_cards) < 5:
            win_rate = self.second_round(state,hand_cards, board_cards)
            if win_rate == 0:
                win_rate =self.get_win_prob(state, playerid,hand_cards, board_cards,player_num)
                print("win Rate: ", win_rate)
        else:
            win_rate =self.get_win_prob(state, playerid,hand_cards, board_cards,player_num)
        #return ACTION(action_table.RAISE, 50) # all in
        #print "win Rate:{}".format(win_rate)
        action = Action(state)
        min_raise = max(state.community_state.lastraise * 2, state.community_state.bigblind) 
        raise_upper = state.player_states[playerid].stack / 4

        if win_rate > 0.95:
            return action.AllIn(playerid)
        elif win_rate > 0.75:
            raise_amount = state.community_state.to_call + int(state.player_states[playerid].stack / 5)
            return action.Raise(raise_upper, min_raise, raise_amount)
        elif win_rate > 0.65:
            raise_amount = state.community_state.to_call + int(state.player_states[playerid].stack / 15)
            return action.Raise(raise_upper, min_raise, raise_amount)
        if win_rate > 0.45:
            return action.Call()
        else:
            return action.Fold()

    def getReload(self, state):
        '''return `True` if reload is needed under state, otherwise `False`'''
        if self.reload_left > 0:
            self.reload_left -= 1
            return True
        else:
            return False

        if win_rate > 0.5:
            return action.Call()
        else:
            return action.Fold()

    def getReload(self, state):
        '''return `True` if reload is needed under state, otherwise `False`'''
        if self.reload_left > 0:
            self.reload_left -= 1
            return True
        else:
            return False



