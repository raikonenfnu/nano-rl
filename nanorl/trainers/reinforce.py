from collections import deque
import torch
import numpy as np

class reinforceTrainer(object):
    def __init__(self, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.gamma = gamma
        self.print_every = print_every

    def train(self, env, policy, optimizer):
        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, self.n_episodes+1):
            saved_log_probs = []
            rewards = []
            state = env.reset()[0]
            for t in range(self.max_t):
                action, log_prob = policy.act(state)
                saved_log_probs.append(log_prob)
                state, reward, done, _, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break 
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            discounts = [self.gamma**i for i in range(len(rewards)+1)]
            R = sum([a*b for a,b in zip(discounts, rewards)])
            
            policy_loss = []
            for log_prob in saved_log_probs:
                policy_loss.append(-log_prob * R)
            policy_loss = torch.cat(policy_loss).sum()
            
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if i_episode % self.print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque)>=195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
                break
        return scores

