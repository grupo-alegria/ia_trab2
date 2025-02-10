from itertools import product
import socket
import struct
import json
from multiprocessing import Manager, Pool, cpu_count
from cnn import CNN  # Importa a classe CNN de um arquivo ou módulo 'cnn'
import torch  # Importa o PyTorch, que é utilizado para criar redes neurais e treinar modelos
from torchvision import datasets  # Importa os datasets da biblioteca torchvision
from torchvision.transforms import v2  # Importa as transformações de imagens da torchvision
import time
import threading

# Configurações do Multicast
MULTICAST_GROUP = '224.1.1.1'  # Mesmo grupo do servidor
PORT = 5009                    # Mesma porta do servidor
NAME = "objeto1"

# Define as transformações que serão aplicadas nas imagens de treino e teste
def define_transforms(height, width):
    data_transforms = {
        'train': v2.Compose([  # Define as transformações para o conjunto de treino
            v2.Resize((height, width)),  # Redimensiona as imagens para as dimensões (height, width)
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # Converte a imagem e define o tipo de dado como float32
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normaliza a imagem usando valores de média e desvio padrão para imagens pré-treinadas
        ]),
        'test': v2.Compose([  # Define as transformações para o conjunto de teste
            v2.Resize((height, width)),  # Redimensiona as imagens
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # Converte e define o tipo de dado como float32
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normaliza com os mesmos parâmetros
        ])
    }
    return data_transforms  # Retorna as transformações para treino e teste


# Função para carregar as imagens dos diretórios de treino, validação e teste
def read_images(data_transforms):
    train_data = datasets.ImageFolder('./data/resumido/train/', transform=data_transforms['train'])  # Carrega as imagens de treino
    validation_data = datasets.ImageFolder('./data/resumido/validation/', transform=data_transforms['test'])  # Carrega as imagens de validação
    test_data = datasets.ImageFolder('./data/resumido/test/', transform=data_transforms['test'])  # Carrega as imagens de teste
    return train_data, validation_data, test_data  # Retorna os conjuntos de dados


def train_model_parallel(args):
    model_name, num_epochs, learning_rate, weight_decay, replicacoes, train_data, validation_data, test_data = args
    cnn = CNN(train_data, validation_data, test_data, 8)
    inicio = time.time()
    acc_media, rep_max = cnn.create_and_train_cnn(model_name, num_epochs, learning_rate, weight_decay, replicacoes)
    fim = time.time()
    duracao = fim - inicio
    return model_name, num_epochs, learning_rate, weight_decay, acc_media, rep_max, duracao

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

    print(f"{NAME} multicast aguardando mensagens...")
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

                    inicio_sistema = time.time()

                    replicacoes = mensagem.get("replicacoes", "Parâmetro 'replicações' não encontrado")
                    model_names = mensagem.get("model_names", "Parâmetro 'model_names' não encontrado")
                    epochs = mensagem.get("epochs", "Parâmetro 'epochs' não encontrado")
                    learning_rates = mensagem.get("learning_rates", "Parâmetro 'learning_rates' não encontrado")
                    weight_decays = mensagem.get("weight_decays", "Parâmetro 'weight_decays' não encontrado")
                    num_tasks = mensagem.get("tasks", "Parâmetro 'tasks' não encontrado")

                    num_nucleos = max(1, cpu_count() // 2)
                    print(f"Usando {num_nucleos} núcleos para treinamento paralelo.")

                    # Define as dimensões das imagens (224x224) e aplica as transformações
                    data_transforms = define_transforms(224, 224)
                    train_data, validation_data, test_data = read_images(data_transforms)

                    # Gerenciador para compartilhamento de dados entre processos
                    with Manager() as manager:
                        shared_train_data = manager.list(train_data)
                        shared_validation_data = manager.list(validation_data)
                        shared_test_data = manager.list(test_data)

                        parameter_combinations = list(product(model_names, epochs, learning_rates, weight_decays, [replicacoes]))
                        tasks = [(model_name, num_epochs, learning_rate, weight_decay, replicacoes, 
                                shared_train_data, shared_validation_data, shared_test_data)
                            for model_name, num_epochs, learning_rate, weight_decay, replicacoes in parameter_combinations]
                    
                        tasks_selected = tasks[:num_tasks]
                        print(f"tasks a serem processadas: {len(tasks_selected)}.")

                        # Multiprocessamento usando Pool
                        with Pool(processes=num_nucleos) as pool:
                            results = pool.map(train_model_parallel, tasks_selected)
                            
                    # Cria um lock para proteger a seção crítica
                    lock = threading.Lock()
                    treinamentos_str = ""
                    melhorReplicacaoJSON = ""
                    melhorAcuracia = 0
                    # Processar resultados
                    for model_name, num_epochs, learning_rate, weight_decay, acc_media, rep_max, duracao in results:
                        resultado = ( 
                            f"Modelo: {model_name}\n"
                            f"Épocas: {num_epochs}\n"
                            f"Learning Rate: {learning_rate}\n"
                            f"Weight Decay: {weight_decay}\n"
                            f"Acurácia Média: {acc_media}\n"
                            f"Melhor replicação: {rep_max}\n"
                            f"Tempo: {duracao:.2f} segundos\n"
                            "---------------------------------\n"
                        )
                        with lock:
                            if melhorAcuracia < acc_media:
                                melhorReplicacaoJSON = resultado
                                melhorAcuracia = acc_media
                            treinamentos_str += resultado

                    fim_sistema = time.time()
                    treinamentos_str += f"Tempo total para o sistema Centralizado Multiprocesso: {fim_sistema - inicio_sistema:.2f} segundos"
                    print(treinamentos_str)
                    treinamentos = ""
                    treinamentos = f"Melhor conjunto de parametros do {NAME}: \n{melhorReplicacaoJSON}\n=============================================={treinamentos_str}"
                    with open("distribuido_obj_01.txt", "w") as arquivo:
                        arquivo.write(treinamentos)
                        
                    respond_client("processamento concluido!")
                    
