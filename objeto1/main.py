# Must run 'python.exe -m Pyro5.nameserver' command before run server

from cnn import CNN  # Importa a classe CNN de um arquivo ou módulo 'cnn'
import torch  # Importa o PyTorch, que é utilizado para criar redes neurais e treinar modelos
from torchvision import datasets  # Importa os datasets da biblioteca torchvision
from torchvision.transforms import v2  # Importa as transformações de imagens da torchvision
import time
from itertools import product
import os
from multiprocessing import Pool, cpu_count
import cnn
import json
import Pyro5.api
import concurrent.futures


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

@Pyro5.api.expose
class AI_trainer(object):

    def __init__(self, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        
    def process_tasks(self, client_proxy):
        """
        Processa as tarefas enviadas pelo cliente.
        Cada thread solicita um conjunto de parâmetros do cliente, executa o treinamento e retorna os resultados.
        """
        while True:
            try:
                # Solicita parâmetros do cliente
                params = client_proxy.request_params()
                if not params:  # Finaliza se o cliente não tiver mais tarefas
                    break

                print(f"Thread processando parâmetros: {params}")

                # Processa os parâmetros de treinamento
                model_name, num_epochs, learning_rate, weight_decay = params
                resultados = self.train(model_name, num_epochs, learning_rate, weight_decay, 2)

                # Retorna os resultados ao cliente
                client_proxy.receive_results(resultados)

            except Exception as e:
                print(f"Erro ao processar tarefa: {e}")
                break
            
    @Pyro5.api.expose            
    def initPool(self, client_uri):
        """
        Inicializa o pool de threads para processar tarefas do cliente.
        """
        client_proxy = Pyro5.api.Proxy(client_uri)
        num_nucleos = cpu_count()
        print(f"Inicializando pool de threads com {num_nucleos} núcleos...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_nucleos) as executor:
            futures = [executor.submit(self.process_tasks, client_proxy) for _ in range(num_nucleos)]
            concurrent.futures.wait(futures)

    def train(self, model_name, num_epochs, learning_rate, weight_decay, replications):
        cnn = CNN(self.train_data, self.validation_data, self.test_data, 8)  # Instancia o modelo CNN
        inicio = time.time()  # Marca o início do tempo de treinamento

        acc_media, rep_max = cnn.create_and_train_cnn(model_name, num_epochs, learning_rate, weight_decay, replications)

        fim = time.time()  # Marca o final do tempo de treinamento
        duracao = fim - inicio  # Calcula a duração do treinamento

        resultados = {
            "Parametro de modelo": model_name,
            "Parametro de quantidade de epocas": num_epochs,
            "Parametro de Learning Rate": learning_rate,
            "Parametro de Weight Decay": weight_decay,
            "Acuracia Media": acc_media,
            "Melhor replicacao": rep_max,
            "Tempo": duracao,
            "Unidade de Tempo": "segundos"
        }

        # Adiciona o dicionário para a estrutura JSON
        print(json.dumps(resultados, indent=4))
        resultadosJson = json.dumps(resultados, indent=4)

        return resultadosJson


if __name__ == '__main__':
    print("Running AI_trainer RMI Server...")

    # Defina as transformações e os dados aqui, ou passe como argumento para a classe AI_trainer
    data_transforms = define_transforms(224, 224)
    train_data, validation_data, test_data = read_images(data_transforms)

    # Crie a instância do servidor com os dados de treinamento, validação e teste
    ai_trainer_instance = AI_trainer(train_data, validation_data, test_data)

    # Inicie o servidor Pyro5
    daemon = Pyro5.server.Daemon()
    print("\tFinding the Name Server...")
    ns = Pyro5.api.locate_ns()

    print("\tCreating AI_trainer object URI...")
    uri = daemon.register(ai_trainer_instance)

    print("\tRegistering AI_trainer object at Name Server...")
    ns.register("example.ai_trainer", uri)

    print("\tWaiting method calls...")
    daemon.requestLoop()