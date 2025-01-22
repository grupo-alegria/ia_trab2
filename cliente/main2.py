import Pyro5.api
from itertools import product
import json

class Client:
    def __init__(self, server_uri):
        self.server = Pyro5.api.Proxy(server_uri)
        self.parameters_queue = []  # Fila de parâmetros

    def add_parameters(self, parameters):
        self.parameters_queue.append(parameters)

    def get_next_parameters(self):
        if self.parameters_queue:
            return self.parameters_queue.pop(0)
        return "exit"  # Envia "exit" quando não há mais parâmetros

    def receive_results(self, results):
        print("Resultados recebidos:")
        print(json.dumps(results, indent=4))

if __name__ == '__main__':
    server_uri = "PYRONAME:example.ai_trainer"
    client = Client(server_uri)

    model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
    epochs = [10, 20]  # Número de épocas para treinamento
    learning_rates = [0.001, 0.0001, 0.00001]  # Taxas de aprendizado
    weight_decays = [0, 0.0001]  # Decaimento de peso

    # Gera todas as combinações possíveis de parâmetros
    parameter_combinations_list = list(product(model_names, epochs, learning_rates, weight_decays))

    # Transforma as tuplas em listas (opcional, se precisa estritamente de listas e não de tuplas)
    parameter_combinations = [list(combination) for combination in parameter_combinations_list]

    # Inicia o treinamento paralelo no servidor
    trainer_proxy = Pyro5.api.Proxy(server_uri)
    trainer_proxy.train_parallel("PYRONAME:example.client")
