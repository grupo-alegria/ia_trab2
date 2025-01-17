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
class AITrainer:
    def __init__(self, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

    def train(self, model_name, num_epochs, learning_rate, weight_decay, replications):
        cnn = CNN(self.train_data, self.validation_data, self.test_data, 8)
        start_time = time.time()
        
        acc_avg, best_replication = cnn.create_and_train_cnn(model_name, num_epochs, learning_rate, weight_decay, replications)

        duration = time.time() - start_time
        return {
            "model": model_name,
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "accuracy": acc_avg,
            "best_replication": best_replication,
            "time_seconds": duration
        }

if __name__ == '__main__':
    print("Starting AITrainer RMI Server...")

    # Configuração inicial
    transform = define_transforms(224, 224)
    train_data, validation_data, test_data = read_images(transform)

    # Instância do objeto remoto
    trainer_instance = AITrainer(train_data, validation_data, test_data)

    # Configuração do servidor Pyro5
    daemon = Pyro5.server.Daemon()
    print("\tConnecting to Name Server...")
    ns = Pyro5.api.locate_ns()

    print("\tRegistering AITrainer object...")
    uri = daemon.register(trainer_instance)
    ns.register("example.ai_trainer", uri)

    print("\tServer ready. Waiting for calls...")
    daemon.requestLoop()