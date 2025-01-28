# Must run 'python.exe -m Pyro5.nameserver' command before run server

import json
from multiprocessing import cpu_count
import queue
import threading
import time
import Pyro5.api
from itertools import product
from torchvision import datasets
from torchvision.transforms import v2
import torch
from cnn import CNN

# Define as transformações aplicadas nas imagens
def define_transforms(height, width):
    transform = v2.Compose([
        v2.Resize((height, width)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

# Função para carregar as imagens
def read_images(transform):
    train_data = datasets.ImageFolder('./data/resumido/train/', transform=transform)
    validation_data = datasets.ImageFolder('./data/resumido/validation/', transform=transform)
    test_data = datasets.ImageFolder('./data/resumido/test/', transform=transform)
    return train_data, validation_data, test_data

@Pyro5.api.expose
class AI_Trainer2:
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

        return json.dumps(resultados, indent=4)

if __name__ == '__main__':
    print("Starting AITrainer2 RMI Server...")

    # Configuração inicial
    transform = define_transforms(224, 224)
    train_data, validation_data, test_data = read_images(transform)

    # Instância do objeto remoto
    ai_trainer2_instance = AI_Trainer2(train_data, validation_data, test_data)

    # Configuração do servidor Pyro5
    daemon = Pyro5.server.Daemon()
    print("\tConnecting to Name Server...")
    ns = Pyro5.api.locate_ns()

    print("\tRegistering AITrainer2 object...")
    uri = daemon.register(ai_trainer2_instance)
    ns.register("node.ai_trainer2", uri)

    print("\tServer ready. Waiting for calls...")
    daemon.requestLoop()