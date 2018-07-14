from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table
from treys import Card, Deck, Evaluator
import math
import random
import json
import numpy as np

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
    
    

class MontecarloModel():
    def __init__(self):
        self._nothing = "test"
        self.reload_left = 2
        self.model = {"seed":831}

    def batchTrainModel(self):
        return

    def onlineTrainModel(self):
        return

    def saveModel(self, path):
        return

    def loadModel(self, path):
        return

    def get_win_prob(self,state, playerid):
        def get_card_class(card_int_list):
            res = [Card.new(Card.int_to_str(c)) for c in card_int_list if c != -1]
            return res
        
        def win_prob(board, hand):
            evaluator = Evaluator()
            percentage = 1.0 - evaluator.get_five_card_rank_percentage(evaluator.evaluate(board, hand))
            return percentage

        hand_cards = get_card_class(state.player_states[playerid].hand)
        board_cards = get_card_class(state.community_card)
        if len(board_cards) < 3:
            percentage = 0.6 #! calculate percentage
        else:
            percentage = win_prob(board_cards, hand_cards)
        return percentage 

    def takeAction(self, state, playerid):
        win_rate =self.get_win_prob(state, playerid)
        #print("win Rate: ", win_rate)
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
