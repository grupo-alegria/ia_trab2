from cnn import CNN  # Importa a classe CNN de um arquivo ou módulo 'cnn'
import torch  # Importa o PyTorch, que é utilizado para criar redes neurais e treinar modelos
from torchvision import datasets  # Importa os datasets da biblioteca torchvision
from torchvision.transforms import v2  # Importa as transformações de imagens da torchvision
import time
from itertools import product

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


if __name__ == '__main__':
    # Define as dimensões das imagens (224x224) e aplica as transformações
    data_transforms = define_transforms(224, 224)
    
    # Carrega os dados de treino, validação e teste com as transformações aplicadas
    train_data, validation_data, test_data = read_images(data_transforms)
    
    # Cria uma instância do modelo CNN com os dados carregados e o número de classes (8)
    cnn = CNN(train_data, validation_data, test_data, 8)
    
    # Configurações para treinamento do modelo
    replicacoes = 2  # Número de repetições para treinar o modelo
    model_names = ['alexnet', 'mobilenet_v3_large', 'mobilenet_v3_small', 'resnet18', 'resnet101', 'vgg11', 'vgg19']
    epochs = 1 # [10, 20]  # Número de épocas para treinamento
    learning_rates = [0.001, 0.0001, 0.00001]  # Taxas de aprendizado
    weight_decays = [0, 0.0001]  # Decaimento de peso

    # Gera todas as combinações possíveis de parâmetros
    parameter_combinations = product(model_names, epochs, learning_rates, weight_decays)

    # Itera sobre cada combinação de parâmetros
    for model_name, num_epochs, learning_rate, weight_decay in parameter_combinations:
        inicio = time.time()  # Marca o início do tempo de treinamento

        # Treina a CNN utilizando os parâmetros definidos
        acc_media, rep_max = cnn.create_and_train_cnn(model_name, num_epochs, learning_rate, weight_decay, replicacoes)

        fim = time.time()  # Marca o final do tempo de treinamento
        duracao = fim - inicio  # Calcula a duração do treinamento

        # Exibe os resultados do treinamento
        print(f"\nParâmetro de modelo: {model_name}")
        print(f"Parâmetro de quantidade de épocas: {num_epochs}")
        print(f"Parâmetro de Learning Rate: {learning_rate}")
        print(f"Parâmetro de Weight Decay: {weight_decay}")
        print(f"Acurácia Média: {acc_media}")
        print(f"Melhor replicação: {rep_max}")
        print(f"Tempo: {duracao:.2f} segundos\n")
        print("-----------------------------------")
