# Must run 'python.exe -m Pyro5.nameserver' command before run server

import queue
import threading
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
class AI_Trainer1:
    def __init__(self, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.task_queue = queue.Queue()
        self.resultadosTotais = []  # Lista compartilhada para resultados
        self.lock = threading.Lock()  # Lock para acesso seguro

    def get_cpu_count(self):
        """
        Retorna o número de CPUs disponíveis no nó.
        """
        return cpu_count()

    def add_tasks(self, tasks):
        """
        Adiciona uma lista de tarefas à fila do nó.
        """
        for task in tasks:
            self.task_queue.put(task)

    def worker(self):
        """
        Worker que processa tarefas da fila.
        """
        while not self.task_queue.empty():
            try:
                # Obtem uma tarefa da fila
                model_name, num_epochs, learning_rate, weight_decay, replications = self.task_queue.get(timeout=1)
                print(f"Nó processando: {model_name}")

                # Processa a tarefa usando a função train
                resultado = self.train(model_name, num_epochs, learning_rate, weight_decay, replications)
                print(f"Tarefa concluída: {resultado}")

                # Adiciona o resultado à lista compartilhada com Lock
                with self.lock:
                    self.resultadosTotais.append(resultado)

            except queue.Empty:
                break
            finally:
                self.task_queue.task_done()
                
    @Pyro5.api.oneway
    def start_processing(self):
        """
        Inicializa threads para processar as tarefas.
        """
        num_threads = min(cpu_count(), self.task_queue.qsize())
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=self.worker)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()  # Aguarda as threads finalizarem

        print(self.resultadosTotais)

        # Retorna todos os resultados processados
        return self.resultadosTotais

    def train(self, model_name, num_epochs, learning_rate, weight_decay, replications):
        """
        Função de treinamento.
        """
        cnn = CNN(self.train_data, self.validation_data, self.test_data, 8)
        inicio = time.time()

        acc_media, rep_max = cnn.create_and_train_cnn(model_name, num_epochs, learning_rate, weight_decay, replications)

        fim = time.time()
        duracao = fim - inicio

        # Criando a string do resultado
        resultados = ( 
            f"Modelo: {model_name}\n"
            f"Épocas: {num_epochs}\n"
            f"Learning Rate: {learning_rate}\n"
            f"Weight Decay: {weight_decay}\n"
            f"Acurácia Média: {acc_media}\n"
            f"Melhor replicação: {rep_max}\n"
            f"Tempo: {duracao:.2f} segundos\n"
            "---------------------------------\n"
        )


        # Abrindo o arquivo em modo 'a' para adicionar novos resultados sem sobrescrever os antigos
        with open("distribuido_obj_01.txt", "a") as arquivo:
            arquivo.write(resultados)
            arquivo.write(f"Tempo total para o Sistema Distribuído: {duracao:.2f} segundos\n\n")

        return json.dumps(resultados, indent=4)


if __name__ == '__main__':
    print("Running AITrainer1 RMI Server...")

    # Defina as transformações e os dados aqui, ou passe como argumento para a classe AITrainer1
    data_transforms = define_transforms(224, 224)
    train_data, validation_data, test_data = read_images(data_transforms)

    # Crie a instância do servidor com os dados de treinamento, validação e teste
    ai_trainer1_instance = AI_Trainer1(train_data, validation_data, test_data)

    # Inicie o servidor Pyro5
    daemon = Pyro5.server.Daemon()
    print("\tFinding the Name Server...")
    ns = Pyro5.api.locate_ns()

    print("\tCreating AITrainer1 object URI...")
    uri = daemon.register(ai_trainer1_instance)

    print("\tRegistering AITrainer1 object at Name Server...")
    ns.register("node.ai_trainer1", uri)

    print("\tWaiting method calls...")
    daemon.requestLoop()