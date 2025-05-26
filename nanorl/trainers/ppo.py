from collections import deque
import torch
from torch.distributions import Categorical
import numpy as np

class PPOTrainer(object):
    def __init__(self, n_episodes=1000, max_t=1000, gamma=0.995, print_every=100, sgd_epoch=2, device=torch.device("cpu")):
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.gamma = gamma
        self.print_every = print_every
        self.sgd_epoch = sgd_epoch
        self.device = device
        self.epsilon = 0.1

    def train(self, env, policy, optimizer):
        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, self.n_episodes+1):
            old_probs = []
            rewards = []
            states = []
            actions = []
            state = env.reset()[0]
            for t in range(self.max_t):
                with torch.no_grad():
                    probs = policy.forward(torch.Tensor(state).to(self.device).reshape(1,-1))
                    m = Categorical(probs)
                    action = m.sample()
                    old_probs.append(probs[0, action.item()])
                new_state, reward, done, _, _ = env.step(action.item())
                states.append(torch.Tensor(state))
                state = new_state
                actions.append(action.item())
                rewards.append(reward)
                if done:
                    break

            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            discounts = torch.Tensor([self.gamma**i for i in range(len(rewards))]).to(self.device)
            rewards = torch.Tensor(rewards).to(self.device)
            rewards *= discounts
            rewards_future = torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=-1),dims=(0,))
            rewards_std = torch.std(rewards_future) + 1.0e-10
            rewards_mean = torch.mean(rewards_future)
            rewards_future = (rewards_future - rewards_mean) /rewards_std

            states = torch.vstack(states).to(self.device)
            old_probs = torch.hstack(old_probs).to(self.device)
            step_range = [i for i in range(len(actions))]
            # Reuse trajectory (sgd_epoch) num of times.
            for _ in range(self.sgd_epoch):
                new_probs = policy.forward(states)[step_range, actions]
                ratio = new_probs / old_probs
                loss = (ratio * rewards_future)
                clipped_loss = rewards_future * torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
                policy_loss = -torch.mean(torch.min(loss, clipped_loss))
                
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()

            # the clipping parameter reduces as time goes on
            self.epsilon*=.999
            
            # the regulation term also reduces
            # this reduces exploration in later runs
            # beta*=.995

            if i_episode % self.print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque)>=195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
                break
        return scores

