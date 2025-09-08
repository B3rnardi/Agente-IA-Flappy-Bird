import gymnasium as gym
from stable_baselines3 import PPO
from flappy_env import FlappyBirdEnv
import os

MODELS_DIR = "models/PPO_stable" # <-- Garanta que o nome da pasta está correto

def play_latest_model():
    """
    Carrega o modelo mais recente da pasta e o executa com visualização.
    """
    try:
        if not os.path.exists(MODELS_DIR) or not os.listdir(MODELS_DIR):
            print(f"Erro: A pasta '{MODELS_DIR}' está vazia ou não existe.")
            print("Por favor, treine um modelo primeiro.")
            return

        saved_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.zip')]
        if not saved_models:
            print(f"Nenhum modelo .zip encontrado em '{MODELS_DIR}'.")
            return
            
        latest_model_name = max(saved_models, key=lambda x: int(x.split('.')[0]))
        latest_model_path = os.path.join(MODELS_DIR, latest_model_name)
        
        print(f"Carregando o melhor modelo: {latest_model_path}")
        model = PPO.load(latest_model_path)
        
        env = FlappyBirdEnv()
        obs, info = env.reset()
        
        print("\n--- Iniciando Demonstração (Feche a janela para sair) ---")
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                print(f"Fim de jogo! Pontuação final: {info['score']}")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nDemonstração interrompida.")
    finally:
        if 'env' in locals():
            env.close()

if __name__ == '__main__':
    play_latest_model()