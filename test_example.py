import holdem
import agent
import hashlib

SERVER_URI = r"ws://allhands2018-beta.dev.spn.a1q7.net:3001"


# Only One Player 
#  name = str(hashlib.md5("831_01".encode('utf-8')).hexdigest())
name="831_01"
print("Player Name ... {}".format(name))
model = agent.idiotModel()
client_player = holdem.ClientPlayer(SERVER_URI, name, model, debug=True)

client_player.doListen()

'''
# Mulriple Player
from multiprocessing import Pool
name_list=["831_01","831_02","831_03","831_04","831_05"]
'''