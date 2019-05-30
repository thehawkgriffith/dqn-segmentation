import numpy as np
import torch 

class Environment:

    def __init__(self, sen_vectors, win_size, labels, pos_rew, neg_rew):
        self.sen_vectors = sen_vectors
        self.pointer = 0
        self.win_size = win_size
        self.labels = labels
        self.pos_rew = pos_rew
        self.neg_rew = neg_rew
        self.observation_shape = (self.win_size, self.sen_vectors.shape[1])
        self.n_actions = self.win_size

    def step(self, action):
        curr_rew = self.neg_rew
        if action == None:
            deletion = False
            indices = [x for x in range(self.pointer, self.pointer + self.win_size)]
            for index in indices:
                if index in self.labels:
                    deletion = True
            if not deletion:
                curr_rew = self.pos_rew
        else:
            hyp_seg = self.pointer + action
            if hyp_seg in self.labels:
                curr_rew = self.pos_rew
        return self.next_obs(), curr_rew, self.is_done()
    
    def reset(self):
        self.pointer = 0
        vecs = self.sen_vectors[self.pointer:self.pointer+self.win_size]
        vecs = torch.FloatTensor(vecs)
        return vecs

    def is_done(self):
        if self.pointer == len(self.sen_vectors) - self.win_size:
            return True
        return False

    def next_obs(self):
        self.pointer += 1
        vecs = self.sen_vectors[self.pointer:self.pointer+self.win_size]
        vecs = torch.FloatTensor(vecs)
        return vecs

    def action_sample(self):
        return np.random.choice([i for i in range(self.win_size)])

def make(data_path, labels_path, win_size, pos_rew, neg_rew):
    sen_vectors = np.load(data_path)
    labels = np.load(labels_path)
    segments = [label[0] for label in labels]
    env = Environment(sen_vectors, win_size, segments, pos_rew, neg_rew)
    return env