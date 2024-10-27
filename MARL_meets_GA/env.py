import numpy as np
import matplotlib.pyplot as plt
import random
from agent import Agent

class CooperativeCollectionEnv:
    def __init__(self, agents, grid_size=8, num_items=5, num_obstacles=2):
        self.grid_size = grid_size
        self.num_items = num_items
        self.agents = agents
        self.items = []
        self.num_obstacles = num_obstacles  # タプルで直接指定した障害物の座標

    def reset(self):
        self._set_obstacles()
        for agent in self.agents:
            agent.reset_position()
        self.items = [self._random_position() for _ in range(self.num_items)]
        self.steps = 0
        self.done = False
        return [agent.get_partial_observation(self.items, self.agents, self.obstacles) for agent in self.agents]

    def _set_obstacles(self):
        self.obstacles = [(random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)) for _ in range(self.num_obstacles)]
    
    def _random_position(self):
        position = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        # 障害物と重複しないように位置を設定
        while tuple(position) in self.obstacles:
            position = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        return position

    def step(self, actions):
        rewards = np.full(len(self.agents), -1)
        
        # 各エージェントの行動（0: 上, 1: 下, 2: 左, 3: 右）を実行
        for i, action in enumerate(actions):
            self.agents[i].move(action, self.obstacles)

            # アイテムの収集をチェック
            for item in self.items:
                if self.agents[i].position == item:
                    rewards[i] = 10
                    self.items.remove(item)  # 収集されたアイテムを削除
                    break  # 1回の行動で1つのアイテムしか収集しない

        # 全アイテムが収集されるとタスク完了
        if not self.items:
            self.done = True

        self.steps += 1
        observations = [agent.get_partial_observation(self.items, self.agents, self.obstacles) for agent in self.agents]
        return observations, rewards, self.done

    def render(self):
        # グリッドの可視化
        grid = np.full((self.grid_size, self.grid_size), ' ')
        for agent in self.agents:
            x, y = agent.position
            grid[y, x] = 'A'
        for item_pos in self.items:
            x, y = item_pos
            grid[y, x] = 'I'
        for obstacle in self.obstacles:
            x, y = obstacle
            grid[y, x] = 'X'  # 障害物を 'X' で表す
        
        print("\n".join("".join(row) for row in grid))
        print("\n" + "-" * 20)
