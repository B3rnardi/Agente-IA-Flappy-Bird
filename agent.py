import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
from model import FlappyBirdAI

# --- Constantes do Agente ---
MAX_MEMORY = 100_000
BATCH_SIZE = 4096
LR = 0.0005 # Learning Rate (Taxa de aprendizado)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon_start = 1.0  # Começa 100% aleatório
        self.epsilon_end = 0.01   # Mínimo de 1% de aleatoriedade
        self.epsilon_decay = 0.9999 # Fator de decaimento a cada jogo
        self.epsilon = self.epsilon_start
        
        self.gamma = 0.99 # Vamos ajustar isso depois
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = FlappyBirdAI(4, 256, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        
    def get_action(self, state):
        """Decide uma ação: aleatória (exploração) ou a melhor (explotação)."""
        final_move = [0, 0] # [não pular, pular]
        
        if random.random() < self.epsilon:
            # Ação aleatória
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            # Ação baseada na previsão do modelo
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

    def get_state(self, game):
        """Pega o estado atual do jogo e o converte em 5 números."""
        pass

    def remember(self, state, action, reward, next_state, done):
        """Armazena uma experiência na memória."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Treina a IA usando um lote de experiências passadas da memória."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Pega uma amostra aleatória
        else:
            mini_sample = self.memory
        
        # Agrupa os dados da amostra
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Treina a IA com base na última ação tomada."""
        self.train_step(state, action, reward, next_state, done)
        
    def train_step(self, state, action, reward, next_state, done):
        """A lógica de treinamento do Q-Learning."""
        # Converte os dados para tensores PyTorch
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1: # Se for um único dado (short memory)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: Pega a predição do modelo para o estado atual
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]: # Se o jogo não acabou
                # Calcula o Q-value futuro usando a equação de Bellman
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        # 2: Calcula a perda (loss) e otimiza
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


    def get_action(self, state):
        """Decide uma ação: aleatória (exploração) ou a melhor (explotação)."""
        # Epsilon-Greedy: exploração vs explotação
        self.epsilon = 80 - self.n_games # O epsilon diminui a cada jogo
        final_move = [0, 0] # [não pular, pular]
        
        if random.randint(0, 200) < self.epsilon:
            # Ação aleatória
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            # Ação baseada na previsão do modelo
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move