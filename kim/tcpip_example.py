import socket
import struct
from queue import Queue
import threading

def tcp_thread(ip, port, queue_send, queue_recv, num_send, num_recv):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, port))
    s.listen(1)
    print('Waiting for connection from Simulink...')
    conn, addr = s.accept()
    print('Connected with Simulink ', addr)

    while True:
        try:
            # Send
            data_send = queue_send.get()
            struct_format = '!{}d'.format(num_send)
            data_send = struct.pack(struct_format, data_send)
            conn.send(data_send)

            # Receive
            data_recv = conn.recv(num_recv*8)
            if not data_recv:
                break
            try:
                struct_format = '!{}d'.format(num_recv)
                unpacked_data = struct.unpack(struct_format, data_recv)
                queue_recv.put(unpacked_data)
            except:
                queue_recv.put(False)            
        except Exception as e:
            print('Connection killed. ', e)
            break
    
    print("Connection closed.", port)
    queue_recv.put(False)
    conn.close()

if __name__ == "__main__":
    ip = '127.0.0.1'
    port = 65432
    queue_send = Queue()
    queue_recv = Queue()
    queue_send.put(1.2)
    num_send = 1
    num_recv = 1

    t = threading.Thread(target=tcp_thread, daemon=True, args=(ip, port, queue_send, queue_recv, num_send, num_recv))
    t.start()

    while True:
        data_recv = queue_recv.get()
        print(data_recv)
        queue_send.put(data_recv[0])
        if data_recv == False:
            print('End')
            break