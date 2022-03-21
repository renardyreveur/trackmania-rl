import socket
import json

HOST = "127.0.0.1"
PORT = 20222

# Create a TCP socket server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind socket to address, and enable server to accept connections
s.bind((HOST, PORT))
s.listen()

# Accept a connection
conn, addr = s.accept()
print(f"{addr} connected!")

while True:
    data = conn.recv(1024)
    result = json.loads(data.decode('utf-8'))
    print(result['race_finished'], type(result['race_finished']))
    print(result['front_speed'], type(result['front_speed']))

s.close()


"""
Reward = Positive Speed, Negative Distance, Negative Time, Big positive Finish, Big positive Checkpoint, Negative on Impact

+ WR trajectory (best time)

"""