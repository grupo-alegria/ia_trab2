import socket
import struct
import json
from multiprocessing import Pool, cpu_count

# Configurações do Multicast
MULTICAST_GROUP = '224.1.1.1'  # Mesmo grupo do servidor
PORT = 5009                    # Mesma porta do servidor
NAME = "objeto1"

def get_cpu_count():
    """
    Retorna o número de CPUs disponíveis no nó.
    """
    return cpu_count()

def respond_client(message):
    response = {
    "sender": NAME,
    "receiver": "client",
    "response": message
    }

    response_str = json.dumps(response)
    
    response_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    response_sock.sendto(response_str.encode(), (MULTICAST_GROUP, PORT))

if __name__ == '__main__':
    # Criar socket UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', PORT))  # Escuta em todas as interfaces de rede na porta definida

    # Configurar o socket para participar do grupo multicast
    mreq = struct.pack("4sl", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    print("Objeto1 multicast aguardando mensagens...")
    while True:
        data, addr = sock.recvfrom(1024)  # Recebe mensagens do grupo multicast
        mensagem = json.loads(data.decode())
        sender = mensagem.get("sender", "Chave 'sender' não encontrada")  # Obtém o valor da chave "sender"
        receiver = mensagem.get("receiver", "Chave 'receiver' não encontrada")  # Obtém o valor da chave "sender"
        action = mensagem.get("action", "Chave 'action' não encontrada")  # Obtém o valor da chave "sender"
        # print(f"Mensagem JSON recebida de {addr}: {mensagem}")
        # print(f"Sender: {sender}")

        if sender == "client" : 
            
            if receiver == "objeto1":

                if action == "get_cpu_count":
                    print('Obtendo o número de CPUs')
                    numnucleos = get_cpu_count()
                    print(numnucleos," nucleos.")
                    respond_client(numnucleos)
                
                elif action == "start_process":
                    print('Processamento iniciado')
                    print(mensagem)
                    
