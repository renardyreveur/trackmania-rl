/*
 * author: Renardy
 * v1.0
 */


// Create a new socket
auto sock = Net::Socket();
bool connected = false;


// Socket Send Data
bool WriteData(Net::Socket@ s, string jsonstring)
{
	if (!s.WriteRaw(jsonstring))
	{
		print("Socket Write Failed!");
		return false;
	}
	return true;
}


// Runs every frame
void Update(float dt)
{
	if (connected){
		string data = Data::Collect();
		if (data != ""){
			WriteData(sock, data);
		}
	}
}


// Entrypoint
void Main()
{
	// Initiate a socket to given address and port
	if (!sock.Connect("127.0.0.1", 20222)){
		print("Socket connection failed!");
		return;
	}

	print("Connecting to Host...");

	// Wait until socket is fully connected.
	while (!sock.CanWrite()) {
		yield();
	}

	print("Connected!");
	connected = true;

}