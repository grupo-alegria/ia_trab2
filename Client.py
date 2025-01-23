import Pyro5

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