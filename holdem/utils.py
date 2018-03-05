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
from treys import Card
from collections import namedtuple

PLAYER_STATE = namedtuple('player_state', ['emptyplayer', 'seat', 'stack', 'playing_hand', 'handrank', 'playedthisround', 'betting', 'isallin', 'lastsidepot', 'reloadCount', 'hand'])
'''
emptyplayer, (boolean), True seat is empty, False is not
seat, (number or string), when play inside gym, this is a seat id, when connect to TM server, this would be player's name
stack, (number), players current stack (TM's chips)
playing_hand, (boolean), player is playing current hand (playing current cycle) (1=True, 0=False(not playing)
handrank, (number), treys.Evaluator.evaluate(hand, community)
playedthisround, (boolean), whether player is plaed this round (1 cycle has 4 rounds)
betting, (number), how much amount players have betting in this cycle
isallin, (boolean), 0 not all in, 1 all in
lastsidepot, (numer), resolve when someone all in  <NOT USING NOW>
reloadCount, (number), only used when TM's version <ONLY TM USED>
hand, (list), information about two card <IN TREYS FORMAT>
'''
COMMUNITY_STATE = namedtuple('community_state', ['button', 'smallblind', 'bigblind', 'totalpot', 'lastraise', 'call_price', 'to_call', 'current_player'])
'''
button, (id), the id of bigblind {e.g. button(id=0), smallblind(id=1), bigblind(id=2) }
smallblind, (number), the current small blind amount
bigblind, (number), the current big blind amount 
totalpot, (number), the current total amount in the community pot 
lastraise, (number), the last posted raise amount
call_price, (number), minimum required raise amount, (acuumulate all round)
to_call, (number), the amount required to call, (current round)
current_player, (id), the id of current player
'''
STATE = namedtuple('state', ['player_states', 'community_state', 'community_card'])

ACTION = namedtuple('action', ['action', 'amount'])

class action_table():
    CHECK = 0
    CALL = 1
    RAISE = 2
    FOLD = 3
    NA = 0

def format_action(player, action):
    color = False
    try:
        from termcolor import colored
        # for mac, linux: http://pypi.python.org/pypi/termcolor
        # can use for windows: http://pypi.python.org/pypi/colorama
        color = True
    except ImportError:
        pass
    [aid, raise_amt] = action
    if aid == action_table.CHECK:
        text = '_ check'
        if color:
            text = colored(text, 'white')
        return text
    if aid == action_table.CALL:
        text = '- call, current bet: {}'.format(player.currentbet)
        if color:
            text = colored(text, 'yellow')
        return text
    if aid == action_table.RAISE:
        text = '^ raise, current bet: {}'.format(raise_amt)
        if color:
            text = colored(text, 'green')
        return text
    if aid == action_table.FOLD:
        text = 'x fold'
        if color:
            text = colored(text, 'red')
        return text

def card_to_str(card):
    if card == -1:
        return ''
    return Card.int_to_pretty_str(card)

def card_to_normal_str(card):
    " {'23456789TJQKA'} + {'shdc''} (note: lower case) "
    if card == -1:
        return ''
    return Card.int_to_str(card)

def card_str_to_list(card_str):
    return Card.new(card_str[:1] + card_str[1:].lower())

def hand_to_str(hand):
    output = " "
    for i in range(len(hand)):
        c = hand[i]
        if c == -1:
            if i != len(hand) - 1:
                output += '[  ],'
            else:
                output += '[  ] '
            continue
        if i != len(hand) - 1:
            output += str(Card.int_to_pretty_str(c)) + ','
        else:
            output += str(Card.int_to_pretty_str(c)) + ' '
    return output

def safe_actions(cur_state, n_seats): #  play safe actions, check when no one else has raised, call when raised.
    current_player = cur_state.community_state.current_player
    to_call = cur_state.community_state.to_call
    actions = [[action_table.CHECK, action_table.NA]] * n_seats
    if to_call > 0:
        actions[current_player] = [action_table.CALL, action_table.NA]
    return actions

def model_list_action(cur_state, n_seats, model_list):
    current_player = cur_state.community_state.current_player
    actions = [[action_table.CHECK, action_table.NA]] * n_seats

    model_decision = model_list[current_player].takeAction(cur_state, current_player)
    actions[current_player] = [model_decision.action, model_decision.amount]
    return actions