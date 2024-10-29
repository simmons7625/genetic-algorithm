import numpy as np
import pygame
import random
import torch

class CooperativeCollectionEnv:
    def __init__(self, agents, grid_size=8, num_items=5, num_obstacles=2, reward=10, move_item=False, seed=None):
        self.grid_size = grid_size
        self.num_items = num_items
        self.agents = agents
        self.items = []
        self.num_obstacles = num_obstacles  # タプルで直接指定した障害物の座標
        self.cell_size = 60 # 各セルのピクセルサイズ
        self.reward = reward
        self.move_item = move_item
        self.window_size = self.grid_size * self.cell_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Grid Environment")
        if seed:
            random.seed=seed

    def reset(self):
        self._set_obstacles()
        for agent in self.agents:
            agent.reset_position()
        self.items = [self._random_position() for _ in range(self.num_items)]
        self.steps = 0
        self.done = False
        return torch.stack([torch.tensor(agent.get_partial_observation(self.items, self.agents, self.obstacles), dtype=torch.float) for agent in self.agents])

    def _set_obstacles(self):
        self.obstacles = []
        while len(self.obstacles) < self.num_obstacles:
            obstacle = (random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2))
            # 障害物がエージェントやアイテムの隣接セルを囲まないようにする
            if obstacle in self.obstacles:
                continue

    def _random_position(self):
        position = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        # 障害物と重複しないように位置を設定
        while tuple(position) in self.obstacles:
            position = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        return position

    def step(self, actions):
        rewards = np.full(len(self.agents), 0)
        # 各エージェントの行動（0: 上, 1: 下, 2: 左, 3: 右）を実行
        for i, action in enumerate(actions):
            self.agents[i].move(action, self.obstacles)

            # アイテムの収集をチェック
            for item in self.items:
                if self.agents[i].position == item:
                    rewards[i] = self.reward
                    self.items.remove(item)  # 収集されたアイテムを削除
                    break  # 1回の行動で1つのアイテムしか収集しない
         
        # 全アイテムが収集されるとタスク完了
        if not self.items:
            self.done = True
            pygame.quit()

        # 各itemの移動を実行
        if self.move_item:
            self._move_items()

        self.steps += 1

        # 各エージェントの観測をテンソルとして取得
        observations = torch.stack([torch.tensor(agent.get_partial_observation(self.items, self.agents, self.obstacles), dtype=torch.float) for agent in self.agents])
        return observations, rewards, self.done

    def render(self):
        # 背景を白で塗りつぶし
        self.screen.fill((255, 255, 255))

        # グリッドの各セルを描画
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # グリッドの枠

        # 障害物の描画
        for obstacle in self.obstacles:
            x, y = obstacle
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 0), rect)  # 障害物は黒

        # アイテムの描画
        for item_pos in self.items:
            x, y = item_pos
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 255, 0), rect)  # アイテムは緑

        # エージェントの描画
        for agent in self.agents:
            x, y = agent.position
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 255), rect)  # エージェントは青

        # 画面更新
        pygame.display.flip()

    # itemが動的にエージェントから逃げるよう設定
    def _move_items(self):
        for item_index, item in enumerate(self.items):
            # 現在のアイテムの位置
            item_x, item_y = item
            
            # アイテムの移動可能な方向（上、下、左、右、またはその場に留まる）
            possible_moves = {
                'up': (item_x, max(0, item_y - 1)),
                'down': (item_x, min(self.grid_size - 1, item_y + 1)),
                'left': (max(0, item_x - 1), item_y),
                'right': (min(self.grid_size - 1, item_x + 1), item_y),
                'stay': (item_x, item_y)  # 移動しない選択肢
            }

            # 各移動方向における最も近いエージェントからの距離を計算
            max_distance = -np.inf
            best_move = 'stay'
            for move, (new_x, new_y) in possible_moves.items():
                # グリッド範囲内であり、障害物がない場合のみ移動を考慮
                if (new_x, new_y) in self.obstacles or not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
                    continue

                # 各エージェントからの距離を計算し、最も近いエージェントとの距離を取得
                min_distance_to_agent = min(abs(new_x - agent.position[0]) + abs(new_y - agent.position[1]) for agent in self.agents)

                # 最も遠い移動を選択（最も近いエージェントとの距離を最大化）
                if min_distance_to_agent > max_distance:
                    max_distance = min_distance_to_agent
                    best_move = move

            # 最適な移動を実行
            self.items[item_index] = possible_moves[best_move]