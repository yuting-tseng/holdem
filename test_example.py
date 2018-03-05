import holdem
import agent

SERVER_URI = r"ws://allhands2018-beta.dev.spn.a1q7.net:3001"

# Only One Player
name="831_01"
model = agent.idiotModel()
client_player = holdem.ClientPlayer(SERVER_URI, name, model, debug=True, playing_live=True)

client_player.doListen()

'''
# TODO: Using multiprocess to create multiple clent player
from multiprocessing import Pool
name_list=["831_01","831_02","831_03","831_04","831_05"]
'''