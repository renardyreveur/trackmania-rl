import json
import socketserver


# TCP Socket Server Handler
def start_sever(work_queue, host, port):
    class TMDataGrabber(socketserver.BaseRequestHandler):
        def handle(self):
            while True:
                data = self.request.recv(1024).strip()
                if data is not None:
                    data = [x for x in data.decode("utf-8").split("\n") if x.startswith("{\"") and x.endswith("e}")]
                    if len(data) > 0:
                        result = json.loads(data[-1])
                        work_queue.put(result)

    with socketserver.TCPServer((host, port), TMDataGrabber) as server:
        print(f"Socket Server: Connected to {server.server_address}")
        server.serve_forever()
