import holdem
import agent

SERVER_URI = r"ws://allhands2018-beta.dev.spn.a1q7.net:3001" # beta
# SERVER_URI = r"ws://allhands2018-training.dev.spn.a1q7.net:3001" # training

# name="Enter Your Name Here"
name="831_01"
model = agent.allRaiseModel()

# while True: # Reconnect after Gameover
client_player = holdem.ClientPlayer(SERVER_URI, name, model, debug=True, playing_live=True)
client_player.doListen()