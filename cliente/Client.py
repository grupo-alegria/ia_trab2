import Pyro5.api
import queue

@Pyro5.api.expose
class Client:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.load_tasks()

    def load_tasks(self):
        """
        Carrega as tarefas de parâmetros na fila.
        """
        tasks = [
            ["alexnet", 10, 0.001, 0.01],
            ["mobilenet_v3_large", 20, 0.0001, 0.005],
            ["vgg11", 15, 0.0005, 0.002]
        ]
        for task in tasks:
            self.task_queue.put(task)

    def request_params(self):
        """
        Retorna o próximo conjunto de parâmetros ou None se a fila estiver vazia.
        """
        if not self.task_queue.empty():
            return self.task_queue.get()
        return None

    def receive_results(self, results):
        """
        Recebe e exibe os resultados do servidor.
        """
        print("Resultados recebidos:")
        print(results)


if __name__ == "__main__":
    # Cria e registra o cliente no daemon
    client = Client()
    with Pyro5.api.Daemon() as daemon:
        client_uri = daemon.register(client)  # Registra o cliente e obtém o URI
        print(f"Cliente registrado com URI: {client_uri}")

        # Conecta ao servidor e passa o URI do cliente
        with Pyro5.api.Proxy("PYRONAME:example.ai_trainer") as ai_trainer:
            try:
                ai_trainer.initPool(client_uri)  # Passa o URI registrado
            except Exception as e:
                print(f"Erro durante a execução do sistema distribuído: {e}")

        # Inicia o loop do daemon para manter o cliente disponível
        print("Cliente aguardando chamadas do servidor...")
        daemon.requestLoop()
