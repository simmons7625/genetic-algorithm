import csv
from datetime import datetime
import os
import torch
from tqdm import tqdm

from agent import Agent, Mixer
from env import CooperativeCollectionEnv
from ExperienceReplay import ExperienceReplay

# 学習パラメータ
train_config = {
    'max_steps': 1100000,
    'max_cycle':1000,
    'save_interval': 100000,
    'update_interval': 10,
    'update_tau': 0.01,
    'gamma': 0.99,
    'mixer':'AIQatten',
    'mixer_lr': 1e-4,
    'q_lr': 1e-4,
    'eps_start': 1,
    'eps_end': 0.05,
    'eps_decay': 200000,
    'vision_range': 1,
    'grid_size':10, # 10 * 10
    'num_items': 2,
    'num_agents':3,
    'num_obstacles': 10,
    'item_reward': 10,
    'load_model': False,
    'replay_buffer_capacity': 100000,
    'exploration_steps': 100000,
    'batch_size': 32
}

def now_str(str_format='%Y%m%d%H%M'):
    return datetime.now().strftime(str_format)

def write_options(csv_path, train_config):
    with open(csv_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Option Name', 'Option Value'])
        writer.writeheader()
        writer.writerows({'Option Name': k, 'Option Value': v} for k, v in train_config.items())

def initialize_result_directory():
    result_dir = f'./result/{now_str("%Y%m%d_%H%M%S")}'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f'{result_dir}/model', exist_ok=True)
    return result_dir

def save_model(mixer, agents, result_dir, step):
    model_path = os.path.join(result_dir, f'model/episode_{step}_')
    mixer.save_network(model_path)
    for i, agent in enumerate(agents):
        agent.save_network(path=model_path + f'{i}_')

def log_results(log_path, metrics, values):
    with open(log_path, 'a') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(metrics)
        writer.writerow(values)
        
def select_actions(agents, obs, eps):
    actions = []
    for i in range(len(agents)):
        action = agents[i].action(obs, eps, i)
        actions.append(action)
    return actions

def train(train_config=train_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _train(train_config, device)

def _train(train_config, device):
    result_dir = initialize_result_directory()
    step_log_path = os.path.join(result_dir, 'step_log.csv')
    ep_log_path = os.path.join(result_dir, 'ep_log.csv')
    write_options(os.path.join(result_dir, 'options.csv'), train_config)
    
    replay_buffer = ExperienceReplay(train_config['replay_buffer_capacity'])
    
    save_interval = train_config['save_interval']
    batch_size = train_config['batch_size']
    num_agents = train_config['num_agents']
    vision_range = train_config['vision_range']
    agents = [Agent(agent_id=i, grid_size=train_config['grid_size'], vision_range=vision_range, lr=train_config['q_lr'], gamma=train_config['gamma'], device=device) for i in range(num_agents)]
    mixer = Mixer(device=device, gamma=train_config['gamma'], lr=train_config['mixer_lr'], model=train_config['mixer'], state_dim=agents[0].state_dim)
    if train_config['load_model']:
        model_path = train_config['model_path']
        mixer.load_network(model_path)
        for i in range(num_agents):
            agents[i].load_network(model_path + f'{i}_')
    
    step = 0
    with tqdm(total=train_config['max_steps'], desc="Training Steps") as pbar:
        while step < train_config['max_steps']:
            env = CooperativeCollectionEnv(
                grid_size=train_config['grid_size'], 
                agents=agents,
                num_items=train_config['num_items'],
                num_obstacles=train_config['num_obstacles'],
                reward=train_config['item_reward'],
                move_item=True
            )
            observations = env.reset()
            score = 0
            losses = []
            
            for cycle in range(train_config['max_cycle']):
                # エピソード終了判定
                if step >= train_config['max_steps']:
                    break
                
                # εの設定
                if step <= train_config['exploration_steps']:
                    eps = train_config['eps_start']
                else:
                    eps = max(train_config['eps_end'], train_config['eps_start'] * (1 - step / train_config['eps_decay']))
                    env.render()
                actions = select_actions(agents=agents, obs=observations, eps=eps)
                next_observations, rewards, _ = env.step(actions)
                if rewards.sum() > 0:
                    reward = rewards.sum()  # 報酬を合計
                else:
                    reward = -0.1 * train_config['item_reward']
                
                replay_buffer.push(observations, actions, reward, next_observations)
                
                # 経験リプレイと訓練
                if step >= train_config['exploration_steps'] and len(replay_buffer.buffer) >= batch_size:
                    obs, act, rwd, next_obs = replay_buffer.sample(batch_size)
                    td_errors = []
                    for i in range(batch_size):
                        td_error = mixer.compute_td_error(rwd[i], obs[i], next_obs[i], act[i], agents)
                        td_errors.append(td_error)
                    loss = mixer.train(td_errors, agents)
                    losses.append(loss.detach().item())
                    
                score += reward
                observations = next_observations
                
                # パラメータの更新
                if step % train_config['update_interval'] == 0:
                    mixer.update_target(tau=train_config['update_tau'])
                    for agent in agents:
                        agent.update_target(tau=train_config['update_tau'])

                # モデルの保存
                if step % save_interval == 0:
                    save_model(mixer, agents, result_dir, step)
                
                log_results(step_log_path, ['reward', 'loss'], [reward, losses[-1] if losses else 'nan'])
                
                step += 1
                pbar.update(1)  # プログレスバーの更新
                
                if env.done:
                    break
            
            # エピソード毎のログ
            log_results(ep_log_path, ['reward', 'loss'], [score, sum(losses) / len(losses) if losses else 'nan'])
            print(f"Step {step} - Score: {score}, Loss: {sum(losses) / len(losses) if losses else 'nan'}")
            
            if step >= train_config['max_steps']:
                break

if __name__ == "__main__":
    train()
