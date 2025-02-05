import asyncio
import socket
import struct
from cnn import CNN  # Importa a classe CNN de um arquivo ou módulo 'cnn'
import torch  # Importa o PyTorch, que é utilizado para criar redes neurais e treinar modelos
from torchvision import datasets  # Importa os datasets da biblioteca torchvision
from torchvision.transforms import v2  # Importa as transformações de imagens da torchvision
import time
from itertools import product
from multiprocessing import Pool, cpu_count, Manager
import json

# Configuração do Multicast
MULTICAST_GROUP = '224.1.1.1'
PORT = 5009
NAME = "client"

# Define as transformações que serão aplicadas nas imagens de treino e teste
def define_transforms(height, width):
    data_transforms = {
        'train': v2.Compose([
            v2.Resize((height, width)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': v2.Compose([
            v2.Resize((height, width)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
    return data_transforms

# Função para carregar as imagens dos diretórios de treino, validação e teste
def read_images(data_transforms):
    train_data = datasets.ImageFolder('./data/resumido/train/', transform=data_transforms['train'])
    validation_data = datasets.ImageFolder('./data/resumido/validation/', transform=data_transforms['test'])
    test_data = datasets.ImageFolder('./data/resumido/test/', transform=data_transforms['test'])
    return train_data, validation_data, test_data

# Função para enviar mensagens multicast
def send_multicast_message(message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.sendto(message.encode(), (MULTICAST_GROUP, PORT))
    
# Função para escutar mensagens multicast
def listen_multicast():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', PORT))
    mreq = struct.pack("4sl", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    #while True:
    data, addr = sock.recvfrom(1024)
    print(f"Recebido de {addr}: {data.decode()}")
    return data.decode()

async def mainDistributed():
    #--------------solicita num cpus do objeto1
    message = {
    "sender": NAME,
    "receiver": "objeto1",
    "action": "get_cpu_count"
    }

    # Convertendo para JSON
    json_message = json.dumps(message, indent=4)
    send_multicast_message(json_message)
    response = listen_multicast()
    response_json = json.loads(response)
    nucleos_objeto_1 = response_json.get("response", "Chave 'response' não encontrada")  # Obtém o valor da chave "response"
    print("Resposta do nó de treino:", response)
    print("nucleos_objeto_1 : ",nucleos_objeto_1)
    
    #---------------solicita num cpus do objeto1
    message = {
    "sender": NAME,
    "receiver": "objeto2",
    "action": "get_cpu_count"
    }

    # Convertendo para JSON
    json_message = json.dumps(message, indent=4)
    send_multicast_message(json_message)
    response = listen_multicast()
    response_json = json.loads(response)
    nucleos_objeto_2 = response_json.get("response", "Chave 'response' não encontrada")  # Obtém o valor da chave "response"
    print("Resposta do objeto2:", response)
    print("nucleos_objeto_2 : ",nucleos_objeto_2)
    
    #---------------calcula o numero de tarefas de cada máquina remota:
    total_cpus = nucleos_objeto_1 + nucleos_objeto_2
    
    
    
    replicacoes = 10  # Número de repetições para treinar o modelo
    model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
    epochs = [5, 10]  # Número de épocas para treinamento
    learning_rates = [0.001, 0.0001, 0.00001]  # Taxas de aprendizado
    weight_decays = [0, 0.0001]  # Decaimento de peso


    parameter_combinations = list(product(model_names, epochs, learning_rates, weight_decays, [replicacoes]))
    tasks = [(model_name, num_epochs, learning_rate, weight_decay, replicacoes)
            for model_name, num_epochs, learning_rate, weight_decay, replicacoes in parameter_combinations]

    tasks1 = int(len(tasks) * (nucleos_objeto_1 / total_cpus))
    tasks2 = len(tasks)-tasks1
    print("tasks: ", len(tasks))
    print("tasks1: ",tasks1)
    print("tasks2: ",tasks2)
    

if __name__ == '__main__':
    inicio_total = time.time()

    print("Escolha o sistema para execução:")
    print("1. Centralizado e um único processo")
    print("2. Centralizado e multiprocesso")
    print("3. Distribuído e multiprocesso")
    escolha = input("Digite o número correspondente ao sistema desejado: ")

    if escolha == "1":
        print("Sistema Centralizado em Único Processo Escolhido.")
    elif escolha == "2":
        print("Sistema Centralizado em Multiprocesso Escolhido.")
    elif escolha == "3":
        print("Sistema Distribuído e Multiprocesso.")
        asyncio.run(mainDistributed())
