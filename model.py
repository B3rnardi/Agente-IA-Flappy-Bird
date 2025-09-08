import torch
import torch.nn as nn

class FlappyBirdAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FlappyBirdAI, self).__init__()
        
        # Rede com duas camadas ocultas para maior capacidade de aprendizado
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Camada 1 (Entrada -> Oculta 1)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), # Camada 2 (Oculta 1 -> Oculta 2)
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)  # Camada 3 (Oculta 2 -> Saída)
        )

    def forward(self, x):
        return self.network(x)