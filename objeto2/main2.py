# Must run 'python.exe -m Pyro5.nameserver' command before run server

from cnn import CNN
import time
import json
import Pyro5.api
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from queue import Queue
from torchvision import datasets
from torchvision.transforms import v2
import torch


# Define as transformações que serão aplicadas nas imagens
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


# Função para carregar os conjuntos de dados
def read_images(data_transforms):
    train_data = datasets.ImageFolder('./data/resumido/train/', transform=data_transforms['train'])
    validation_data = datasets.ImageFolder('./data/resumido/validation/', transform=data_transforms['test'])
    test_data = datasets.ImageFolder('./data/resumido/test/', transform=data_transforms['test'])
    return train_data, validation_data, test_data


@Pyro5.api.expose
class AI_trainer:
    def __init__(self, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.pool = None
        self.task_queue = Queue()

    def start_pool(self):
        if self.pool is None:
            num_threads = multiprocessing.cpu_count()
            self.pool = ThreadPoolExecutor(max_workers=num_threads)
            print(f"Pool de threads iniciada com {num_threads} threads.")
            for _ in range(num_threads):
                self.pool.submit(self.process_tasks)

    def process_tasks(self):
        while True:
            params = self.task_queue.get()  # Espera pela próxima tarefa na fila
            if params == "exit":
                print("Thread recebeu comando de saída. Encerrando apenas esta thread.")
                self.task_queue.task_done()
                break  # Encerra apenas esta thread
            self.train_task(params)
            self.task_queue.task_done()  # Marca a tarefa como concluída

    def train_task(self, params):
        model_name, num_epochs, learning_rate, weight_decay, replications = params

        # Realiza o treinamento do modelo
        cnn = CNN(self.train_data, self.validation_data, self.test_data, 8)
        inicio = time.time()
        acc_media, rep_max = cnn.create_and_train_cnn(model_name, num_epochs, learning_rate, weight_decay, replications)
        fim = time.time()
        duracao = fim - inicio

        resultados = {
            "Parâmetro de modelo": model_name,
            "Parâmetro de quantidade de épocas": num_epochs,
            "Parâmetro de Learning Rate": learning_rate,
            "Parâmetro de Weight Decay": weight_decay,
            "Acurácia Média": acc_media,
            "Melhor replicação": rep_max,
            "Tempo": duracao,
            "Unidade de Tempo": "segundos"
        }

        print(json.dumps(resultados, indent=4))
        resultadosJson = json.dumps(resultados, indent=4)
        return resultadosJson

    def submit_task(self, params):
        """
        Adiciona uma tarefa à fila.
        """
        self.task_queue.put(params)

    def train_parallel(self, param_list):
        """
        Método chamado via RMI. Recebe uma lista de parâmetros para treinamento.
        """
        if not param_list:
            raise ValueError("A lista de parâmetros está vazia.")

        self.start_pool()  # Garante que a pool está ativa
        for params in param_list:
            self.submit_task(params)  # Adiciona os parâmetros à fila


if __name__ == '__main__':
    print("Running AI_trainer RMI Server...")

    data_transforms = define_transforms(224, 224)
    train_data, validation_data, test_data = read_images(data_transforms)

    ai_trainer_instance = AI_trainer(train_data, validation_data, test_data)

    daemon = Pyro5.server.Daemon()
    print("\tFinding the Name Server...")
    ns = Pyro5.api.locate_ns()

    print("\tCreating AI_trainer object URI...")
    uri = daemon.register(ai_trainer_instance)

    print("\tRegistering AI_trainer object at Name Server...")
    ns.register("example.ai_trainer", uri)

    print("\tWaiting for method calls...")
    daemon.requestLoop()
