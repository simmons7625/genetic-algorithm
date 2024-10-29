import numpy as np
import torch
import torch.optim as optim
import copy
from model import *
import random

class Agent: 
    def __init__(self, agent_id, grid_size, vision_range=1, lr=1e-3, gamma=0.99, device='cpu', ):
        self.model = DQN().to(device) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.target_model = copy.deepcopy(self.model)
        self.gamma = gamma
        self.device = device
        self.agent_id = agent_id
        self.position = [0, 0]
        self.grid_size = grid_size
        self.vision_range = vision_range

    def reset_position(self):
        self.position = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]

    def get_partial_observation(self, items, agents, obstacles):
        # 視界範囲のリストを作成
        view_size = 2 * self.vision_range + 1
        observation = np.zeros((view_size, view_size, 3), dtype=int)
        
        # 中心のエージェント位置を (0, 0) として設定
        center = self.vision_range

        # 視界範囲の境界を計算
        min_x = max(0, self.position[0] - self.vision_range)
        max_x = min(self.grid_size - 1, self.position[0] + self.vision_range)
        min_y = max(0, self.position[1] - self.vision_range)
        max_y = min(self.grid_size - 1, self.position[1] + self.vision_range)

        # アイテムの相対座標を視界内に表示
        for item in items:
            if min_x <= item[0] <= max_x and min_y <= item[1] <= max_y:
                rel_x = item[0] - self.position[0] + center
                rel_y = item[1] - self.position[1] + center
                observation[rel_y][rel_x][0] = 1  # アイテムの存在を示す

        # 他のエージェントの相対座標を視界内に表示
        for agent in agents:
            if agent.agent_id != self.agent_id and min_x <= agent.position[0] <= max_x and min_y <= agent.position[1] <= max_y:
                rel_x = agent.position[0] - self.position[0] + center
                rel_y = agent.position[1] - self.position[1] + center
                observation[rel_y][rel_x][1] = 1  # 他のエージェントの存在を示す
                    
        # 障害物の相対座標を視界内に表示
        for obstacle in obstacles:
            if min_x <= obstacle[0] <= max_x and min_y <= obstacle[1] <= max_y:
                rel_x = obstacle[0] - self.position[0] + center
                rel_y = obstacle[1] - self.position[1] + center
                observation[rel_y][rel_x][2] = 1  # 障害物の存在を示す

        # エージェント自身の位置を削除し、フラット化
        observation = observation.reshape((view_size ** 2, 3))
        observation = np.delete(observation, center * view_size + center, axis=0).flatten()  # 中心位置を削除
        position = np.array([self.position[0], self.position[1]])
        observation = np.concatenate([position, observation], axis=-1)
        return torch.tensor(observation, device=self.device, dtype=torch.float)

    def move(self, action, obstacles):
        # 次の位置を計算
        next_position = self.position.copy()
        if action == 0:  # 上
            next_position[1] -= 1
        elif action == 1:  # 下
            next_position[1] += 1
        elif action == 2:  # 左
            next_position[0] -= 1
        elif action == 3:  # 右
            next_position[0] += 1

        # 次の位置がグリッド内であり、かつ障害物がない場合にのみ移動
        if (
            0 <= next_position[0] < self.grid_size and
            0 <= next_position[1] < self.grid_size and
            tuple(next_position) not in obstacles
        ):
            self.position = next_position
    
    def action(self, observations, eps, id):
        if random.random() < eps:
                action = random.randint(0, 3)
        else:
            action_values = self.model(observations, id)
            action = action_values.argmax().item()

        return action
    
    def update_target(self, tau=0.01):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
    def save_network(self, path):
        torch.save(self.model.state_dict(), path + 'learner.pth')
    
    def load_network(self, path):
        self.model.load_state_dict(torch.load(path + 'learner.pth'))
    
class Mixer:
    def __init__(self, device='cpu', model='AIQatten', gamma=0.99, lr=1e-3, gru=False):
        if model == 'AIQatten':
            self.mixer = AIQatten().to(device)
        if model == 'Qatten':
            self.mixer = Qatten().to(device)
        if model == 'QMIX':
            self.mixer = QMIX().to(device)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.optimizer = optim.Adam(self.mixer.parameters(), lr=lr)
        self.gamma = gamma  # 割引率 
        self.gru = gru
    
    def compute_td_error(
        self, reward, features, next_features, actions, learners):
        
        """ TD 誤差の計算 """
        action_values = []
        for i in range(len(learners)):
            action_value = learners[i].model(features, i)
            action_values.append(action_value)
        
        # 現在のQ値を計算
        current_q = self.mixer(
            features, actions, action_values, len(learners), target=False)
        
        next_action_values = []
        for i in range(len(learners)):
            next_action_value = learners[i].model(features, i)
            next_action_values.append(next_action_value)
        
        # 次のQ値を計算
        next_q = self.target_mixer(
            next_features, actions, next_action_values, len(learners), target=True)
        
        # TD誤差を計算
        target_q = reward + self.gamma * next_q.detach()
        td_error = (target_q - current_q) ** 2
        return td_error
    
    def train(self, td_errors, learners):
        td_errors = torch.stack(td_errors)
        loss = td_errors.mean()
        # オプティマイザの更新
        for learner in learners:
            learner.optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for learner in learners:
            learner.optimizer.step()
        return loss
    
    def update_target(self, tau=0.01):
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
    def save_network(self, path):
        torch.save(self.mixer.state_dict(), path + 'mixer.pth')
    
    def load_network(self, path):
        self.mixer.load_state_dict(torch.load(path + 'mixer.pth'))