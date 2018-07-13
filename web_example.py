import holdem
import agent

#SERVER_URI = "ws://poker-training.vtr.trendnet.org:3001/" # beta
SERVER_URI = "ws://poker-dev.wrs.club:3001/" # training

# name="Enter Your Name Here"
name="Beth"
model = agent.MontecarloModel()

# while True: # Reconnect after Gameover
client_player = holdem.ClientPlayer(SERVER_URI, name, model, debug=True, playing_live=True)
client_player.doListen()
