import numpy as np
import torch
import torch.nn as nn
import dqn
import time
import collections
import torch.optim as optim
from environment import make
import pickle

MEAN_REWARD_BOUND = 10
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 50000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 0.02
EPSILON_FINAL = 0.02

Experience = collections.namedtuple(
    'Experience',
    field_names=[
        'state',
        'action',
        'reward',
        'done',
        'new_state'
    ]
)


class ExperienceBuffer:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return np.array([t.numpy() for t in states]), np.array(actions), np.array(rewards, np.float32), np.array(dones, np.uint8), np.array([t.numpy() for t in next_states])


class Agent:

    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
    
    def play_step(self, net, epsilon=0.0, device='gpu'):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_sample()
        else:
            state_a = np.array(self.state, copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        new_state, reward, is_done = self.env.step(action)
        self.total_reward += reward
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device='cpu'):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    state_action_values = net(states_v).squeeze(1).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).squeeze(1).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default=True, action='store_true', 
    # help='Enable cuda')
    # parser.add_argument("--env", default=DEFAULT_ENV_NAME, 
    # help='Name of the environment, default = ' + DEFAULT_ENV_NAME)
    # parser.add_argument('--reward', type=float, default=MEAN_REWARD_BOUND, 
    # help='Mean reward boundary for stopping of training, default = %.2f'%(MEAN_REWARD_BOUND))
    # args = parser.parse_args()
    # device = torch.device('cuda' if args.cuda else 'cpu')
    device = torch.device('cuda')
    env = make('glovedatarms.npy', 'labels2.npy', 4, 1, -1)
    net = dqn.DQN(env.observation_shape, env.n_actions).to(device)
    tgt_net = dqn.DQN(env.observation_shape, env.n_actions).to(device)
    print(net)
    net.load_state_dict(torch.load('best.dat'))
    buffer = ExperienceBuffer(REPLAY_SIZE)
    file = open('replay_mem.obj', 'rb')
    buffer = pickle.load(file)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx/EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame)/(time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d passes, mean reward %.3f, eps %.2f, speed %.2f f/s"%(frame_idx, len(total_rewards), mean_reward, epsilon, speed))
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), 'best.dat')
                filehandler = open('replay_mem.obj', 'wb')
                pickle.dump(buffer, filehandler)
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved"%(best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > best_mean_reward:
                print("Solved in %d frames!"%frame_idx)
                break
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

if __name__ == '__main__':
    main()