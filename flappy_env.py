import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from flappy_game import FlappyBirdGame # Importamos o jogo que já criamos

class FlappyBirdEnv(gym.Env):
    """
    Esta classe é a "porta USB" (interface Gymnasium) para o nosso jogo Flappy Bird.
    """
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        
        # O jogo em si
        self.game = FlappyBirdGame()
        
        # 1. Definir o Espaço de Ações
        # 2 ações possíveis: 0 (não fazer nada) e 1 (pular)
        self.action_space = spaces.Discrete(2)
        
        # 2. Definir o Espaço de Observação (o "estado")
        # O estado tem 4 números (posição Y, velocidade Y, dist X cano, dist Y cano)
        # Box para valores contínuos, definindo os limites mínimo e máximo.
        low = np.array([-1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
    def _get_obs(self):
        """Função auxiliar para pegar o estado do jogo no formato certo."""
        screen_width = self.game.screen.get_width()
        screen_height = self.game.screen.get_height()
        bird_rect = self.game.bird_rect
        next_pipe = self.game._get_next_pipe()
        if next_pipe:
            dist_to_pipe_x = next_pipe.rect.left - bird_rect.right
            dist_to_bottom_opening = next_pipe.rect.top - bird_rect.bottom
        else:
            dist_to_pipe_x = screen_width
            dist_to_bottom_opening = 0
        state = [
            bird_rect.centery / screen_height,
            self.game.bird_velocity / 10,
            dist_to_pipe_x / screen_width,
            dist_to_bottom_opening / screen_height,
        ]
        return np.array(state, dtype=np.float32)

    def _get_info(self):
        """Retorna informações adicionais (não usadas para o treinamento)."""
        return {"score": self.game.score}

    def reset(self, seed=None, options=None):
        """Reinicia o ambiente e retorna a observação inicial."""
        super().reset(seed=seed)
        self.game.reset()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """Executa uma ação no ambiente."""
        action_vec = [0, 0]
        action_vec[action] = 1
        
        reward, game_over, score = self.game.play_step(action_vec)
        
        observation = self._get_obs()
        info = self._get_info()
        
        # A interface do Gym requer `terminated` e `truncated`
        terminated = game_over
        truncated = False # Usado se o jogo tem um limite de tempo
        
        return observation, reward, terminated, truncated, info

    def render(self):
        """Mostra a parte gráfica do jogo."""
        self.game._update_ui()
        self.game.clock.tick(30)

    def close(self):
        """Fecha o ambiente."""
        pygame.quit()