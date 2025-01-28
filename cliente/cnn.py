import numpy as np  # Importa a biblioteca numpy, usada para operações matemáticas e manipulação de arrays.
import torch  # Importa a biblioteca principal do PyTorch.
from torch import nn, optim  # Importa o módulo de redes neurais (nn) e otimização (optim) do PyTorch.
from torch.utils import data  # Importa o utilitário de data do PyTorch, necessário para carregar os dados.
from torchvision import models  # Importa o módulo de modelos da biblioteca torchvision, que contém modelos pré-treinados.

class CNN:  
    def __init__(self, train_data, validation_data, test_data, batch_size):
        # Inicializa a classe com dados de treino, validação e teste, além do tamanho do lote (batch_size).
    
        self.train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        # Cria um DataLoader para o conjunto de treino, usando o batch_size e embaralhando os dados.
        
        self.validation_loader = data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
        # Cria um DataLoader para o conjunto de validação, sem embaralhamento dos dados.
        
        self.test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        # Cria um DataLoader para o conjunto de teste, sem embaralhamento dos dados.
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define que o treinamento e as operações ocorrerão na CPU (pode ser alterado para "cuda" se houver GPU disponível).

    def create_and_train_cnn(self, model_name, num_epochs, learning_rate, weight_decay, replicacoes):
        # Função principal que cria e treina o modelo várias vezes (definido por 'replicacoes') e calcula a média da acurácia.
        
        soma = 0  # Inicializa a variável para somar as acurácias de todas as replicações.
        acc_max = 0  # Inicializa a variável para armazenar a melhor acurácia obtida.
        
        for i in range(0, replicacoes):  # Loop para executar a replicação do treinamento.
            print(f"Replicação {i + 1} de {replicacoes} de {model_name}, sob {num_epochs}, {learning_rate}, {weight_decay} iniciada.")
            model = self.create_model(model_name)  # Cria o modelo de rede neural com base no nome especificado.
            optimizerSGD = self.create_optimizer(model, learning_rate, weight_decay)  # Cria o otimizador SGD com a taxa de aprendizado e o peso de decaimento fornecidos.
            criterionCEL = self.create_criterion()  # Cria a função de perda (CrossEntropyLoss).
            
            # Chama a função para treinar o modelo no conjunto de treino.
            self.train_model(model, self.train_loader, optimizerSGD, criterionCEL, model_name, num_epochs, learning_rate, weight_decay, i)
            
            acc = self.evaluate_model(model, self.validation_loader)  # Avalia a acurácia do modelo no conjunto de validação.
            
            soma += acc  # Soma a acurácia obtida nesta replicação à variável 'soma'.
            
            if acc > acc_max:  # Se a acurácia da replicação atual for maior que a melhor acurácia obtida até agora...
                acc_max = acc  # Atualiza a melhor acurácia.
                iter_acc_max = i  # Armazena o índice da replicação com a melhor acurácia.

        # Retorna a acurácia média das replicações e o índice da replicação com a melhor acurácia.
        return soma / replicacoes, iter_acc_max

    def create_model(self, model_name):
        # Função que cria o modelo com base no nome fornecido.        
        if (model_name == 'VGG11'):  # Se o nome do modelo for 'VGG11'...
            model = models.vgg11(weights='DEFAULT')  # Carrega o modelo VGG11 pré-treinado.
            for param in model.parameters():
                param.requires_grad = False  # Congela os parâmetros para não serem atualizados durante o treinamento.
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)  # Substitui a última camada para ter 2 saídas (classificação binária).
            return model  # Retorna o modelo modificado.
        elif model_name == 'VGG19':
            model = models.vgg19(weights='DEFAULT')
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
            return model
        elif (model_name == 'Alexnet'):  # Se o nome do modelo for 'Alexnet'...
            model = models.alexnet(weights='DEFAULT')  # Carrega o modelo AlexNet pré-treinado.
            for param in model.parameters():
                param.requires_grad = False  # Congela os parâmetros.
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)  # Substitui a última camada para ter 2 saídas.
            return model  # Retorna o modelo modificado.        
        elif model_name == 'resnet18':
            model = models.resnet18(weights='DEFAULT')  # Carrega o modelo ResNet18 pré-treinado
            for param in model.parameters():
                param.requires_grad = False  # Congela os pesos das camadas pré-treinadas
            model.fc = nn.Linear(model.fc.in_features, 2)  # Substitui a última camada para 2 classes
            return model
        elif model_name == 'resnet101':
            model = models.resnet101(weights='DEFAULT')  # Carrega o modelo ResNet101 pré-treinado
            for param in model.parameters():
                param.requires_grad = False  # Congela os pesos das camadas pré-treinadas
            model.fc = nn.Linear(model.fc.in_features, 2)  # Substitui a última camada para 2 classes
            return model
        elif model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(weights='DEFAULT')
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
            return model
        else:  # Se o modelo não for nem VGG11 nem AlexNet, carrega o modelo MobilenetV3Large.
            model = models.mobilenet_v3_large(weights='DEFAULT')  # Carrega o modelo MobilenetV3Large pré-treinado.
            for param in model.parameters():
                param.requires_grad = False  # Congela os parâmetros.
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)  # Substitui a última camada para ter 2 saídas.
            return model  # Retorna o modelo modificado.

    def create_optimizer(self, model, learning_rate, weight_decay):
        # Cria o otimizador para o modelo, utilizando o SGD (Stochastic Gradient Descent).
        
        update = []  # Inicializa uma lista para armazenar os parâmetros a serem otimizados.
        for name, param in model.named_parameters():  # Itera sobre os parâmetros do modelo.
            if param.requires_grad == True:  # Se o parâmetro deve ser atualizado durante o treinamento...
                update.append(param)  # Adiciona o parâmetro à lista de parâmetros a serem atualizados.
        
        # Cria o otimizador SGD com a taxa de aprendizado e o peso de decaimento fornecidos.
        optimizerSGD = optim.SGD(update, lr=learning_rate, weight_decay=weight_decay)
        
        return optimizerSGD  # Retorna o otimizador.

    def create_criterion(self):
        # Cria a função de perda utilizada para o treinamento (CrossEntropyLoss é usada para classificação multi-classe).
        criterionCEL = nn.CrossEntropyLoss()
        return criterionCEL  # Retorna a função de perda.

    def train_model(self, model, train_loader, optimizer, criterion, model_name, num_epochs, learning_rate, weight_decay, replicacao):
        # Função responsável pelo treinamento do modelo durante várias épocas.
        
        model.to(self.device)  # Envia o modelo para o dispositivo (CPU ou GPU).
        min_loss = 100  # Inicializa a variável para armazenar a menor perda durante o treinamento.
        e_measures = []  # Lista para armazenar as medidas de desempenho após cada época.
        
        for i in range(1, num_epochs + 1):  # Loop sobre o número de épocas.
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)  # Treina o modelo por uma época e calcula a perda.
            if (train_loss < min_loss):  # Se a perda for a menor registrada até agora...
                min_loss = train_loss  # Atualiza a menor perda.
                nome_arquivo = f"./modelos/{model_name}_{num_epochs}_{learning_rate}_{weight_decay}_{replicacao}.pth"  # Gera o nome do arquivo para salvar o modelo.
                torch.save(model.state_dict(), nome_arquivo)  # Salva os pesos do modelo no arquivo gerado.

    def train_epoch(self, model, trainLoader, optimizer, criterion):
        # Função que realiza o treinamento por uma única época (passagem por todo o conjunto de treinamento).
        
        model.train()  # Coloca o modelo no modo de treinamento (ativando camadas como dropout, por exemplo).
        losses = []  # Lista para armazenar as perdas de cada mini-lote.
        
        for X, y in trainLoader:  # Itera sobre os mini-lotes no DataLoader.
            X = X.to(self.device)  # Envia os dados de entrada para o dispositivo (CPU ou GPU).
            y = y.to(self.device)  # Envia as etiquetas de saída para o dispositivo.
            
            optimizer.zero_grad()  # Zera os gradientes acumulados dos parâmetros do modelo.
            
            y_pred = model(X)  # Realiza a previsão com o modelo.
            loss = criterion(y_pred, y)  # Calcula a perda (erro) entre as previsões e as etiquetas reais.
            
            loss.backward()  # Calcula os gradientes com relação à perda.
            optimizer.step()  # Atualiza os parâmetros do modelo com base nos gradientes calculados.
            
            losses.append(loss.item())  # Adiciona a perda do mini-lote à lista.
        
        model.eval()  # Coloca o modelo no modo de avaliação (desativa dropout, por exemplo).
        
        return np.mean(losses)  # Retorna a média das perdas de todos os mini-lotes da época.

    def evaluate_model(self, model, loader):
        # Função que avalia a acurácia do modelo no conjunto de dados fornecido (validação ou teste).
        
        total = 0  # Inicializa o contador de exemplos processados.
        correct = 0  # Inicializa o contador de previsões corretas.
        
        for X, y in loader:  # Itera sobre os mini-lotes no DataLoader.
            X, y = X.to(self.device), y.to(self.device)  # Envia os dados e as etiquetas para o dispositivo.
            output = model(X)  # Realiza a previsão com o modelo.
            _, y_pred = torch.max(output, 1)  # Obtém a classe prevista (maior valor da saída do modelo).
            
            total += len(y)  # Incrementa o número total de exemplos.
            correct += (y_pred == y).sum().cpu().data.numpy()  # Conta quantas previsões foram corretas.
        
        acc = correct / total  # Calcula a acurácia como a razão entre previsões corretas e o total de exemplos.
        
        return acc  # Retorna a acurácia do modelo.
