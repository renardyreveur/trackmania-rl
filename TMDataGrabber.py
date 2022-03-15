import socketserver
import json

HOST, PORT = "127.0.0.1", 20222


# TCP Socket Server Handler
def start_sever(work_queue):
    class TMDataGrabber(socketserver.BaseRequestHandler):
        def handle(self):
            while True:
                data = self.request.recv(1024).strip()
                if data is not None:
                    data = data.decode("utf-8").split("\n")
                    result = json.loads(data[0] if len(data) == 1 else data[-2])
                    work_queue.put(result)

    with socketserver.TCPServer((HOST, PORT), TMDataGrabber) as server:
        print(f"Connected to {server.server_address}")
        server.serve_forever()
