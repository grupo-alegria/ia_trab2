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
import threading

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

def listen_multicast_confirmation(message_list, done_event):
    """
    Escuta mensagens multicast (que são JSONs) e as adiciona à lista
    se a key "receiver" for igual a "client".
    Quando duas mensagens válidas forem recebidas, sinaliza o evento.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', PORT))
    
    # Adiciona o socket ao grupo multicast
    mreq = struct.pack("4sl", socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    
    while len(message_list) < 2:
        data, addr = sock.recvfrom(1024)
        try:
            message_str = data.decode()
            message_json = json.loads(message_str)
            # Verifica se a mensagem possui a key "receiver" com o valor "client"
            if message_json.get("receiver") == "client":
                print(f"Recebido de {addr}: {message_str}")
                message_list.append(message_json)
            else:
                print(f"Mensagem de {addr} ignorada: 'receiver' não é 'client'")
        except Exception as e:
            print(f"Erro ao processar mensagem de {addr}: {e}")
    
    # Sinaliza que já foram recebidas duas mensagens válidas
    done_event.set()

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
    print("Resposta do objeto1:", response)
    
    receiver = response_json.get("receiver", "Chave 'receiver' não encontrada")  # Obtém o valor da chave "response"
    sender = response_json.get("sender", "Chave 'sender' não encontrada")  # Obtém o valor da chave "sender"
    if receiver == "client":
        if sender == "objeto1":
            nucleos_objeto_1 = response_json.get("response", "Chave 'response' não encontrada")  # Obtém o valor da chave "response"
            print("nucleos_objeto_1 : ",nucleos_objeto_1)
    
    #---------------solicita num cpus do objeto2
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
    print("Resposta do objeto2:", response)
    
    receiver = response_json.get("receiver", "Chave 'receiver' não encontrada")  # Obtém o valor da chave "response"
    sender = response_json.get("sender", "Chave 'sender' não encontrada")  # Obtém o valor da chave "sender"
    if receiver == "client":
        if sender == "objeto2":
            nucleos_objeto_2 = response_json.get("response", "Chave 'response' não encontrada")  # Obtém o valor da chave "response"
            print("nucleos_objeto_2 : ",nucleos_objeto_2)
            
    
    #---------------calcula o numero de tarefas de cada máquina remota:
    total_cpus = nucleos_objeto_1 + nucleos_objeto_2
    
    
    
    replicacoes = 1  # Número de repetições para treinar o modelo
    # model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
    model_names = ['alexnet', 'mobilenet_v3_large']
    epochs = [2]  # Número de épocas para treinamento
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
    
    ###
    # Lista para armazenar as mensagens recebidas com o confirmação do processamento
    messages = []
    # Evento para sinalizar quando as duas mensagens forem recebidas
    done = threading.Event()
    
    # Cria e inicia a thread que escuta as mensagens multicast
    listener_thread = threading.Thread(target=listen_multicast_confirmation, args=(messages, done))
    listener_thread.start()
    ###
    
    messageObj1 = {
        "replicacoes": replicacoes,
        "model_names": model_names,
        "epochs": epochs,
        "learning_rates": learning_rates,
        "weight_decays": weight_decays,
        "tasks" : tasks1,
        "sender": NAME,
        "receiver": "objeto1",
        "action": "start_process"
    }
    
    print(messageObj1)

    # Convertendo para JSON
    json_message_obj1 = json.dumps(messageObj1, indent=4)
    send_multicast_message(json_message_obj1)
    
    messageObj2 = {
        "replicacoes": replicacoes,
        "model_names": model_names,
        "epochs": epochs,
        "learning_rates": learning_rates,
        "weight_decays": weight_decays,
        "tasks" : tasks2,
        "sender": NAME,
        "receiver": "objeto2",
        "action": "start_process"
    }
    
    print(messageObj2)

    # Convertendo para JSON
    json_message_obj2 = json.dumps(messageObj2, indent=4)
    send_multicast_message(json_message_obj2)
    
    done.wait()
    
    print("Mensagens recebidas:", messages)
    

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
