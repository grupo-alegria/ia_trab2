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
    
def train_model_parallel(args):
    model_name, num_epochs, learning_rate, weight_decay, replicacoes, train_data, validation_data, test_data = args
    cnn = CNN(train_data, validation_data, test_data, 8)
    inicio = time.time()
    acc_media, rep_max = cnn.create_and_train_cnn(model_name, num_epochs, learning_rate, weight_decay, replicacoes)
    fim = time.time()
    duracao = fim - inicio
    return model_name, num_epochs, learning_rate, weight_decay, acc_media, rep_max, duracao

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

def mainDistributed():
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
    epochs = [1]  # Número de épocas para treinamento
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
        inicio_sistema = time.time()  # Início do sistema centralizado único
        print("Sistema Centralizado em Único Processo Escolhido.")
        # Define as dimensões das imagens (224x224) e aplica as transformações
        data_transforms = define_transforms(224, 224)
        
        # Carrega os dados de treino, validação e teste com as transformações aplicadas
        train_data, validation_data, test_data = read_images(data_transforms)
        
        # Cria uma instância do modelo CNN com os dados carregados e o número de classes (8)
        cnn = CNN(train_data, validation_data, test_data, 8)
        
        # Configurações para treinamento do modelo
        replicacoes = 1  # Número de repetições para treinar o modelo
        #model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
        model_names = ['vgg11', 'vgg19', 'mobilenet_v3_large']
        epochs = [1]  # Número de épocas para treinamento
        learning_rates = [0.001, 0.0001]  # Taxas de aprendizado
        weight_decays = [0, 0.0001]  # Decaimento de peso

        # Gera todas as combinações possíveis de parâmetros
        parameter_combinations = product(model_names, epochs, learning_rates, weight_decays)

        #Define a String responsável por registrar os logs dos treinamentos
        treinamentos_str = ""
        melhorReplicacaoJSON = (
                        f"Modelo: {""}\n"
                            f"Épocas: {0}\n"
                            f"Learning Rate: {0}\n"
                            f"Weight Decay: {0}\n"
                            f"Acurácia Média: {0}\n"
                            f"Melhor replicação: {0}\n"
                            f"Tempo: {0:.2f} segundos\n"
                            "---------------------------------\n"
                    )
        melhorAcuracia = 0
        # Itera sobre cada combinação de parâmetros
        for model_name, num_epochs, learning_rate, weight_decay in parameter_combinations:
            inicio = time.time()  # Marca o início do tempo de treinamento

            # Treina a CNN utilizando os parâmetros definidos
            acc_media, rep_max = cnn.create_and_train_cnn(model_name, num_epochs, learning_rate, weight_decay, replicacoes)

            fim = time.time()  # Marca o final do tempo de treinamento
            duracao = fim - inicio  # Calcula a duração do treinamento

            # Exibe os resultados do treinamento                        
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
            if melhorAcuracia < acc_media:
                melhorReplicacaoJSON = resultado
                melhorAcuracia = acc_media
            treinamentos_str = treinamentos_str+resultado
            
        fim_sistema = time.time()
        treinamentos_str = treinamentos_str+f"Tempo total para o sistema Centralizado Único Processo: {fim_sistema - inicio_sistema:.2f} segundos\n"
        print(treinamentos_str)
        treinamentos = ""
        treinamentos = f"Melhor conjunto de parametros: \n{melhorReplicacaoJSON}\n=============================================={treinamentos_str}"        

        with open("centralizado_unico_processo.txt", "w") as arquivo:
            arquivo.write(treinamentos)

    elif escolha == "2":        
        inicio_sistema = time.time()  # Início do sistema centralizado multiprocesso
        print("Sistema Centralizado em Multiprocesso Escolhido.")

        # Obter número de núcleos disponíveis
        #num_nucleos = cpu_count()
        num_nucleos = max(1, cpu_count() // 2)
        print(f"Usando {num_nucleos} núcleos para treinamento paralelo.")

        # Define as dimensões das imagens (224x224) e aplica as transformações
        data_transforms = define_transforms(224, 224)
        train_data, validation_data, test_data = read_images(data_transforms)
    
        # Configurações para treinamento do modelo
        replicacoes = 1
        #model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
        model_names = ['vgg11', 'vgg19', 'mobilenet_v3_large']
        epochs = [1]
        learning_rates = [0.001, 0.0001, 0.00001]
        weight_decays = [0, 0.0001]

        parameter_combinations = list(product(model_names, epochs, learning_rates, weight_decays, [replicacoes]))
        args = [(model_name, num_epochs, learning_rate, weight_decay, replicacoes, train_data, validation_data, test_data)
                for model_name, num_epochs, learning_rate, weight_decay, replicacoes in parameter_combinations]

        # Gerenciador para compartilhamento de dados entre processos
        with Manager() as manager:
            shared_train_data = manager.list(train_data)
            shared_validation_data = manager.list(validation_data)
            shared_test_data = manager.list(test_data)

            args = [(model_name, num_epochs, learning_rate, weight_decay, replicacoes, 
                    shared_train_data, shared_validation_data, shared_test_data)
                    for model_name, num_epochs, learning_rate, weight_decay, replicacoes in parameter_combinations]

            # Multiprocessamento usando memória compartilhada
            with Pool(processes=num_nucleos) as pool:
                results = pool.map(train_model_parallel, args)
            treinamentos_str=""

            melhorReplicacaoJSON = (
                        f"Modelo: {""}\n"
                            f"Épocas: {0}\n"
                            f"Learning Rate: {0}\n"
                            f"Weight Decay: {0}\n"
                            f"Acurácia Média: {0}\n"
                            f"Melhor replicação: {0}\n"
                            f"Tempo: {0:.2f} segundos\n"
                            "---------------------------------\n"
                    )
            melhorAcuracia = 0
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
            if melhorAcuracia < acc_media:
                melhorReplicacaoJSON = resultado
                melhorAcuracia = acc_media

            treinamentos_str = treinamentos_str+resultado

        print(treinamentos_str)
        fim_sistema = time.time()
        treinamentos_str = treinamentos_str+f"Tempo total para o sistema Centralizado Multiprocesso: {fim_sistema - inicio_sistema:.2f} segundos"
        
        treinamentos = ""
        treinamentos = f"Melhor conjunto de parametros: \n{melhorReplicacaoJSON}\n=============================================={treinamentos_str}"
        
        with open("centralizado_multiplos_processos.txt", "w") as arquivo:
            arquivo.write(treinamentos)
        print(treinamentos)

    elif escolha == "3":
        inicio_sistema = time.time()
        print("Sistema Distribuído e Multiprocesso.")
        mainDistributed()
        fim_sistema = time.time()
        treinamentos_str = f"Tempo total para o Sistema Distribuido Multiprocesso: {fim_sistema - inicio_sistema:.2f} segundos"
        print(treinamentos_str)
