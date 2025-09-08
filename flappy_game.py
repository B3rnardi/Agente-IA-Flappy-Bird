import pygame
import sys
import random

pygame.init()
pygame.font.init()

# --- Constantes ---
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
FPS = 300 # FPS alto para treinamento rápido
FONT = pygame.font.SysFont('Consolas', 30, bold=True)

# --- Cores ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 200, 0)

# Classe auxiliar para os canos
class Pipe:
    def __init__(self, x, y, width, height, is_top):
        self.rect = pygame.Rect(x, y, width, height)
        self.passed = False
        self.is_top = is_top

class FlappyBirdGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird AI")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.bird_x = 50
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.bird_rect = pygame.Rect(self.bird_x, self.bird_y, 34, 24)
        self.pipe_list = []
        self.score = 0
        self.game_over = False

    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if action[1] == 1:
            self.bird_velocity = -7

        self.bird_velocity += 0.25
        self.bird_y += self.bird_velocity
        self.bird_rect.y = self.bird_y

        self._spawn_pipes()
        self._move_pipes()

        reward = 0
        next_pipe = self._get_next_pipe()
        if next_pipe and not next_pipe.passed and next_pipe.rect.centerx < self.bird_rect.centerx:
            next_pipe.passed = True
            pair_pipe = self._get_pair_pipe(next_pipe)
            if pair_pipe: pair_pipe.passed = True
            self.score += 1
            reward = 10

        if self._check_collision():
            self.game_over = True
            reward = -1
            return reward, self.game_over, self.score
        
        return reward, self.game_over, self.score

    def _spawn_pipes(self):
        if not self.pipe_list or self.pipe_list[-1].rect.centerx < SCREEN_WIDTH - 200:
            gap = 150
            if len(self.pipe_list) < 10:
                gap = 220
            
            random_pipe_pos = random.randint(200, 380)
            bottom_pipe = Pipe(SCREEN_WIDTH + 10, random_pipe_pos, 70, SCREEN_HEIGHT, is_top=False)
            top_pipe = Pipe(SCREEN_WIDTH + 10, random_pipe_pos - gap - SCREEN_HEIGHT, 70, SCREEN_HEIGHT, is_top=True)
            self.pipe_list.extend([bottom_pipe, top_pipe])
    
    def _move_pipes(self):
        for pipe in self.pipe_list:
            pipe.rect.centerx -= 3
        self.pipe_list = [pipe for pipe in self.pipe_list if pipe.rect.right > 0]

    def _check_collision(self):
        for pipe in self.pipe_list:
            if self.bird_rect.colliderect(pipe.rect):
                return True
        if self.bird_y > SCREEN_HEIGHT - 24 or self.bird_y < 0:
            return True
        return False

    def _update_ui(self):
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, YELLOW, self.bird_rect)
        for pipe in self.pipe_list:
            pygame.draw.rect(self.screen, GREEN, pipe.rect)
        score_text = FONT.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()

    def _get_next_pipe(self):
        for pipe in self.pipe_list:
            if pipe.rect.right > self.bird_rect.left and not pipe.is_top:
                return pipe
        return None
    
    def _get_pair_pipe(self, pipe):
        for p in self.pipe_list:
            if p.rect.left == pipe.rect.left and p.is_top != pipe.is_top:
                return p
        return None