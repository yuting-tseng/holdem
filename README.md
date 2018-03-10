# TrendMicro CoreTech Allhands Holdem Competetion Training Tool 

## known issue
+ sidepot

## OpenAI Gym
[OpenAI Gym 官方網址](https://gym.openai.com/docs/)
> Gym is a toolkit for developing and comparing reinforcement learning algorithms.

> Gym 提供了一個測試 Reinforcement Learning的環境框架, 並且沒有對 Agent做任何假設

另外也可以參考 DeepMind 針對德州撲克的環境[DeepMind Poker](https://github.com/dickreuter/Poker)

## 此版本介紹

這一個版本是從 [wenkesj/holdem](https://github.com/wenkesj/holdem)改寫，主要多增加了以下功能
+ 修改因 openai/gym 在 [commit #836](https://github.com/openai/gym/pull/836) spec change所造成的 crash
+ 新增 `cycle` attribute (發牌一輪為 round, 玩一次為 cycle)
+ 新增 Interface連接 Trend Micro server
+ 新增 agent template (必須提供兩個 method讓 controller呼叫 (controller為溝通 environment與 agent的橋梁)
+ 修改因 全部 player hold所造成的 crash
+ 限制每一 round 同一 player raise 次數上限為 4次 (可透過參數修改) (自動改成 CALL)
+ 修改 to_call為 此 round絕對數值

另外有改寫 [ihendley/treys](https://github.com/ihendley/treys)，這一個repo是改寫自 */deuce為提供 poker相關計算與管理
+ 修改 f-string not supported under python 3.6
+ 修改 serup.py在 windows paltform造成的 encoding issue (cp950 decode error)

### 名詞解釋 (這裡 refactor會影響到 interface暫時先不更改)
1. 局(Episode): 一局遊戲是指參賽的10名玩家進入一張遊戲桌, 持有相等的初始籌碼, 對戰直到本遊戲桌只剩下 不多於半數 玩家勝出, 而其餘玩家因籌碼耗盡而出局為止, 一局由多圈構成
2. 圈(): 一圈是指 dealer button 圍繞牌桌在每個未出局的玩家手上都出現一次為止為一圈,一圈由多輪構成,在一圈過程中大小盲注數額相等,但是下一圈開始時大小盲注翻倍
3. 輪(CYCLE): 一輪遊戲是指每次荷官重新發公共牌和私有牌, 每個玩家按回合進行決策, 直到除了一名玩家之外全部棄牌, 或者5張公共牌完全翻開為止, 決出本輪勝負並清算籌碼, 一輪由多個回合構成
4. 回合(ROUND)): 一回合是指所有玩家依次take action, 稱為一回合, 一個回合中有多個(分別來自各個玩家的)action
5. action(STEP): 一個action是指輪到某一個玩家 call/raise/check/fold/~~bet~~/~~allin~~ 玩家通過 AI 客戶端完成其中一種決策稱之為一個action

## 安裝方法
```sh
# better run under virtualenv
git clone https://github.com/chuchuhao/holdem.git
pip install gym
pip install websocket-client
pip install git+https://github.com/chuchuhao/treys # 若非 windows環境可以直接 pip install treys
```

## 使用方法
- local_example: environment為 gym
- web_example: enviroment為 Trend Micro Server

### Agent需要提供的 interface
Agent必須為一個 class並且提供下面兩個 method
+ `takeAction(self, state, playerid)` return ACTION (namedtupled)
+ `getReload(self, state)` reutrn {True/ False}

### 如何參與 TM holdem
#### Train Phase
- 將下面會介紹的 State Tuple拿出 verorized state 餵給 model吃
- 可以利用 env利用指定 policy來生成 training data做 batch learning
- 可以利用 env並搭配 RL algorithm來做 online training
- 自行撰寫 expert rule

#### Valid Phase
- 可以利用 env來觀察 model行為

#### Test Phase
- 連上 TM Server做測試

## State介紹

:warning: **這裡的 state與 openai/gym裡面所設定的 observation space不完全一樣, 是包裝給 agent用的**

agent會接到一個由 namedtuple所包成的 state, 包含下列三個項目

0. player_states: 長度為此桌 `seat` 座位數的 Tuple
    + 每一個 item為 player_sates:
        + `emptyplayer`, (boolean), 0 seat is empty, 1 is not 表示這一個位子有沒有玩家註冊
        + `seat`, (number), 玩家的 seat number, seat也是 玩家的初始順序, 過程中不會改變
        + `stack`, (number), 玩家剩餘籌碼
        + `playing_hand`, (boolean), 玩家目前有在玩此 cycle
        + `handrank`, (number), 由 treys.Evaluator.evaluate(hand, community), 每一個 round結束後都會計算
        + `playedthisround`, (boolean), 玩家是否已經玩過此 round (1 cycle 有4 rounds)
        + `betting`, (number), 玩家在此 cycle已下注的金額
        + `isallin`, (boolean), 0 not all in, 1 all in
        + `lastsidepot`, (number), resolve when someone all in  <NOT USING NOW> 目前 sidepot相關功能都沒有使用
        + `reloadCount`, (number), <ONLY TM USED> 在 openai/gym中沒有適用到 reload功能
        + `hand`, (list(2)), 長度為 2的<IN TREYS FORMAT> 必須使用 TREYS提供的 API解讀

1. community_state: 這裡所提到的 id = seat number
    + button, (number), the id of bigblind 莊家位置 (順序: 莊家> 小盲 > 大盲)
    + smallblind, (number), the current small blind amount 小盲注籌碼數
    + bigblind, (number), the current big blind amount  大盲注籌碼數
    + totalpot, (number), the current total amount in the community pot 所有人下注的總籌碼數
    + lastraise, (number), the last posted raise amount 最後一個人 raise的籌碼數
    + call_price, (number), 此 round要 call的絕對籌碼數
    + to_call, (number), 此 round要 call的相對籌碼數 (絕對籌碼數 - 此 round已出籌碼數)
    + current_player, (id), the id of current player 目前決策的玩家 id

:warning: 

2. community_cards: 長度為 5的 list, 每一個 item唯一張卡 <IN TREYS FORMAT>

* card -1時代表牌面未公布, 另外 card的值為一 Number須由 TREY解讀

## Action介紹

:warning: **這裡的 action與 openai/gym裡面所設定的 action space不完全一樣, 是包裝給 agent用的**

agent要做出一個 action的時候, 必須丟出一個 ACTION的 namedtuple
+ `action`: 由 action_table() class提供, 這裡與 TM提供的 interface稍有不同, 但會自動轉換
    + `action_table.CHECK` 歲月靜好 (不下注)
    + `action_table.CALL` 跟 (下注, 但不指定金額)
    + `action_table.RAISE` 提高開殺 (下注, 指定親俄)
    + `action_table.FOLD` 放棄 GG (不下注放棄)
+ `amount`: 當地一個選額為 RAISE時, 會查看此一項目


# 以下為原版  wenkesj/holdem版本的 readme

:warning: **This is an experimental API, it will most definitely contain bugs, but that's why you are here!**

```sh
pip install holdem
```

Afaik, this is the first [OpenAI Gym](https://github.com/openai/gym) _No-Limit Texas Hold'em_* (NLTH)
environment written in Python. It's an experiment to build a Gym environment that is synchronous and
can support any number of players but also appeal to the general public that wants to learn how to
"solve" NLTH.

*Python 3 supports arbitrary length integers :money_with_wings:

Right now, this is a work in progress, but I believe the API is mature enough for some preliminary
experiments. Join me in making some interesting progress on multi-agent Gym environments.

# Usage

There is limited documentation at the moment. I'll try to make this less painful to understand.

## `env = holdem.TexasHoldemEnv(n_seats, max_limit=1e9, debug=False)`

Creates a gym environment representation a NLTH Table from the parameters:

+ `n_seats` - number of available players for the current table. No players are initially allocated
  to the table. You must call `env.add_player(seat_id, ...)` to populate the table.
+ `max_limit` - max_limit is used to define the `gym.spaces` API for the class. It does not actually
  determine any NLTH limits; in support of `gym.spaces.Discrete`.
+ `debug` - add debug statements to play, will probably be removed in the future.

### `env.add_player(seat_id, stack=2000)`

Adds a player to the table according to the specified seat (`seat_id`) and the initial amount of
chips allocated to the player's `stack`. If the table does not have enough seats according to the
`n_seats` used by the constructor, a `gym.error.Error` will be raised.

### `(player_states, community_states) = env.reset()`

Calling `env.reset` resets the NLTH table to a new hand state. It does not reset any of the players
stacks, or, reset any of the blinds. New behavior is reserved for a special, future portion of the
API that is yet another feature that is not standard in Gym environments and is a work in progress.

The observation returned is a `tuple` of the following by index:

0. `player_states` - a `tuple` where each entry is `tuple(player_info, player_hand)`, this feature
   can be used to gather all states and hands by `(player_infos, player_hands) = zip(*player_states)`.
   + `player_infos` - is a `list` of `int` features describing the individual player. It contains
     the following by index:
     0. `[0, 1]` - `0` - seat is empty, `1` - seat is not empty.
     1. `[0, n_seats - 1]` - player's id, where they are sitting.
     2. `[0, inf]` - player's current stack.
     3. `[0, 1]` - player is playing the current hand.
     4. `[0, inf]` the player's current handrank according to `treys.Evaluator.evaluate(hand, community)`.
     5. `[0, 1]` - `0` - player has not played this round, `1` - player has played this round.
     6. `[0, 1]` - `0` - player is currently not betting, `1` - player is betting.
     7. `[0, 1]` - `0` - player is currently not all-in, `1` - player is all-in.
     8. `[0, inf]` - player's last sidepot.
   + `player_hands` - is a `list` of `int` features describing the cards in the player's pocket.
     The values are encoded based on the `treys.Card` integer representation.
1. `community_states` - a `tuple(community_infos, community_cards)` where:
   + `community_infos` - a `list` by index:
     0. `[0, n_seats - 1]` - location of the dealer button, where big blind is posted.
     1. `[0, inf]` - the current small blind amount.
     2. `[0, inf]` - the current big blind amount.
     3. `[0, inf]` - the current total amount in the community pot.
     4. `[0, inf]` - the last posted raise amount.
     5. `[0, inf]` - minimum required raise amount, if above 0.
     6. `[0, inf]` - the amount required to call.
     7. `[0, n_seats - 1]` - the current player required to take an action.
   + `community_cards` - is a `list` of `int` features describing the cards in the community.
     The values are encoded based on the `treys.Card` integer representation. There are 5 `int` in
     the list, where `-1` represents that there is no card present.

# Example

```python
import gym
import holdem

def play_out_hand(env, n_seats):
  # reset environment, gather relevant observations
  (player_states, (community_infos, community_cards)) = env.reset()
  (player_infos, player_hands) = zip(*player_states)

  # display the table, cards and all
  env.render(mode='human')

  terminal = False
  while not terminal:
    # play safe actions, check when noone else has raised, call when raised.
    actions = holdem.safe_actions(community_infos, n_seats=n_seats)
    (player_states, (community_infos, community_cards)), rews, terminal, info = env.step(actions)
    env.render(mode='human')

env = gym.make('TexasHoldem-v1') # holdem.TexasHoldemEnv(2)

# start with 2 players
env.add_player(0, stack=2000) # add a player to seat 0 with 2000 "chips"
env.add_player(1, stack=2000) # add another player to seat 1 with 2000 "chips"
# play out a hand
play_out_hand(env, env.n_seats)

# add one more player
env.add_player(2, stack=2000) # add another player to seat 1 with 2000 "chips"
# play out another hand
play_out_hand(env, env.n_seats)
```
