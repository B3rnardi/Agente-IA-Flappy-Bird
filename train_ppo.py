import os
import time
from typing import Callable # <--- NOVA IMPORTAÇÃO

import torch
from stable_baselines3 import PPO

# Certifique-se de que os outros arquivos estão na mesma pasta
from flappy_env import FlappyBirdEnv


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Cria uma função que retorna a taxa de aprendizado diminuindo linearmente.
    `Stable-Baselines3` chama esta função a cada passo, passando o progresso.
    """
    def func(progress_remaining: float) -> float:
        """
        progress_remaining começa em 1.0 e termina em 0.0.
        """
        return progress_remaining * initial_value
    return func


# --- Configuração ---
MODELS_DIR = "models/PPO_stable"
LOG_DIR = "logs/PPO_stable"
TIMESTEPS_PER_SAVE = 50_000 # Salva a cada 50.000 passos

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Instancia o ambiente
env = FlappyBirdEnv()
env.reset()

# Lógica para carregar o último modelo salvo
saved_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.zip')]
if saved_models:
    latest_model_name = max(saved_models, key=lambda x: int(x.split('.')[0]))
    latest_model_path = os.path.join(MODELS_DIR, latest_model_name)
    print(f"Carregando modelo existente: {latest_model_path} para continuar...")
    model = PPO.load(latest_model_path, env=env, tensorboard_log=LOG_DIR)
    start_iteration = int(latest_model_name.split('.')[0]) // TIMESTEPS_PER_SAVE
else:
    print("Nenhum modelo encontrado, começando um novo treinamento.")
    lr_schedule = linear_schedule(0.0003)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=lr_schedule)
    start_iteration = 0
    
# Loop de Treinamento Infinito
iteration = start_iteration
while True:
    iteration += 1
    
    model.learn(total_timesteps=TIMESTEPS_PER_SAVE, reset_num_timesteps=False, tb_log_name="PPO")
    
    total_timesteps_so_far = TIMESTEPS_PER_SAVE * iteration
    model.save(f"{MODELS_DIR}/{total_timesteps_so_far}")
    
    print(f"--- Checkpoint salvo em {total_timesteps_so_far} timesteps ---")

env.close()