from pythonosc import udp_client
import time

client = udp_client.SimpleUDPClient("127.0.0.1", 8020)

for i in range(10):
    client.send_message("/movement/right_hand_x", 0.5)
    print(f"Sent message {i}")
    time.sleep(1)