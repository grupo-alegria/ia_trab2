import socket
import struct

# Configurações do Multicast
MULTICAST_GROUP = '224.1.1.1'  # Mesmo grupo do servidor
PORT = 5009                    # Mesma porta do servidor
NAME = "objeto2"

# Criar socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', PORT))  # Escuta em todas as interfaces de rede na porta definida

# Configurar o socket para participar do grupo multicast
mreq = struct.pack("4sl", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

print("Objeto2 multicast aguardando mensagens...")
while True:
    data, addr = sock.recvfrom(1024)  # Recebe mensagens do grupo multicast
    print(f"Mensagem recebida de {addr}: {data.decode()}")
