import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

class Alice_and_Bob(object):
    def __init__(self, map_size=(13, 9), n=2):
        self.length, self.width = map_size
        self.agent_num = n
        self.generate_map()
        self.map = {'agent':{0: 5, 1: 6}, 'key':{0: 2, 1: 3}, 'lever': 4, 'treasure': 7}
        # print(self.occupancy)
        # print(vec)
    
    def generate_map_obs(self):
        occupancy_obs = np.zeros((self.length, self.width, 3))
        occupancy_obs[self.occupancy==0] = [1,1,1]
        occupancy_obs[self.occupancy==2] = [1,0,1]
        occupancy_obs[self.occupancy==3] = [0,1,1]
        occupancy_obs[self.occupancy==4] = [1,1,0]
        occupancy_obs[self.occupancy==5] = [1,0,0] # agent 1
        occupancy_obs[self.occupancy==6] = [0,0,1] # agent 2
        occupancy_obs[self.occupancy==7] = [0,1,0]
        return occupancy_obs

    def generate_map(self):
        self.occupancy = np.zeros((self.length, self.width))
        
        # enclose the surroundings
        for i in range(self.length):
            self.occupancy[i, 0] = 1
            self.occupancy[i, self.width - 1] = 1
        for i in range(self.width):
            self.occupancy[0, i] = 1
            self.occupancy[self.length -1, i] = 1
        
        # build walls
        for i in range(1,9):
            self.occupancy[i, 4] = 1
        for i in range(1,5):
            self.occupancy[8, i] = 1
        for i in range(5,8):
            self.occupancy[4, i] = 1
        self.occupancy[10, 6] = 1
        self.occupancy[10, 7] = 1
        self.occupancy[11, 6] = 1
        self.secret_wall_pos = [[10, 6], [10, 7], [11, 6]]

        # generate keys and doors
        self.keys_pos = [[3, 5], [11, 1]]
        self.doors_pos = [[8, 2], [4, 6]]
        self.lever_pos = [5, 7]
        self.occupancy[self.keys_pos[0][0]][self.keys_pos[0][1]] = 2
        self.occupancy[self.keys_pos[1][0]][self.keys_pos[1][1]] = 3
        self.occupancy[self.lever_pos[0]][self.lever_pos[1]] = 4
        self.lever_in_use = False

        # initialize agents
        self.agt_pos = [[1, 1], [1, self.width-2]]
        self.occupancy[self.agt_pos[0][0]][self.agt_pos[0][1]] = 5
        self.occupancy[self.agt_pos[1][0]][self.agt_pos[1][1]] = 6

        # initialize treasure
        self.treasure_pos = [11, 7]
        self.occupancy[self.treasure_pos[0]][self.treasure_pos[1]] = 7

        self.occupancy_obs = self.generate_map_obs()

    def reset(self):
        self.generate_map()
        # state = []
        # for i in range(self.agent_num):
        #     state.append(self.get_state(i))
        obs = self.get_obs()
        return obs

    def step(self, action_list):
        reward = 0
        info = [0, 0]
        # agent move
        for i in range(self.agent_num):
            if action_list[i] == 0:  # move up
                if self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]+1] != 1:  # if can move
                    self.agt_pos[i][1] = self.agt_pos[i][1] + 1
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]-1] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = 5+i
                else:
                    reward = reward - 0.1
            elif action_list[i] == 1:  # move down
                if self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]-1] != 1:  # if can move
                    self.agt_pos[i][1] = self.agt_pos[i][1] - 1
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]+1] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = 5+i
                else:
                    reward = reward - 0.1
            elif action_list[i] == 2:  # move left
                if self.occupancy[self.agt_pos[i][0]-1][self.agt_pos[i][1]] != 1:  # if can move
                    self.agt_pos[i][0] = self.agt_pos[i][0] - 1
                    self.occupancy[self.agt_pos[i][0]+1][self.agt_pos[i][1]] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = 5+i
                else:
                    reward = reward - 0.1
            elif action_list[i] == 3:  # move right
                if self.occupancy[self.agt_pos[i][0]+1][self.agt_pos[i][1]] != 1:  # if can move
                    self.agt_pos[i][0] = self.agt_pos[i][0] + 1
                    self.occupancy[self.agt_pos[i][0]-1][self.agt_pos[i][1]] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = 5+i
                else:
                    reward = reward - 0.1

        # check keys
        for i in range(self.agent_num):
            for j in range(len(self.keys_pos)):
                if self.agt_pos[i] == self.keys_pos[j]:
                    self.occupancy[self.doors_pos[j][0]][self.doors_pos[j][1]] = 0 # open the corresponding door
                    info[i] = self.map['key'][j]
        
        # check lever
        if (self.agt_pos[0] == self.lever_pos or self.agt_pos[1] == self.lever_pos) and not self.lever_in_use:
            self.occupancy[10][6] = 0  # clear the wall
            self.occupancy[10][7] = 0  # clear the wall
            self.occupancy[11][6] = 0  # clear the wall
            self.lever_in_use = True
            if self.agt_pos[0] == self.lever_pos:
                info[0] = self.map['lever']
            else: info[1] = self.map['lever']
                
        elif not (self.agt_pos[0] == self.lever_pos or self.agt_pos[1] == self.lever_pos):
            if self.agt_pos[0] not in self.secret_wall_pos and self.agt_pos[1] not in self.secret_wall_pos:
                self.occupancy[10][6] = 1  # reset the wall
                self.occupancy[10][7] = 1  # reset the wall
                self.occupancy[11][6] = 1  # reset the wall
                self.occupancy[self.lever_pos[0]][self.lever_pos[1]] = 4
                self.lever_in_use = False
            elif self.agt_pos[0] in self.secret_wall_pos:
                self.agt_pos[0] = [9, 5] # bounce back to specific pos
                self.occupancy[self.agt_pos[0][0]][self.agt_pos[0][1]] = 5

                self.occupancy[10][6] = 1  # reset the wall
                self.occupancy[10][7] = 1  # reset the wall
                self.occupancy[11][6] = 1  # reset the wall
                self.occupancy[self.lever_pos[0]][self.lever_pos[1]] = 4
                self.lever_in_use = False
                reward -= 10
            elif self.agt_pos[1] in self.secret_wall_pos:
                self.agt_pos[1] = [9, 5] # bounce back to specific pos
                self.occupancy[self.agt_pos[1][0]][self.agt_pos[1][1]] = 6

                self.occupancy[10][6] = 1  # reset the wall
                self.occupancy[10][7] = 1  # reset the wall
                self.occupancy[11][6] = 1  # reset the wall
                self.occupancy[self.lever_pos[0]][self.lever_pos[1]] = 4
                self.lever_in_use = False
                reward -= 10
        
        done = False
        
        # check treasure
        if self.agt_pos[0] == self.treasure_pos or self.agt_pos[1] == self.treasure_pos:
            reward = reward + 100
            if self.agt_pos[0] == self.treasure_pos:
                info[0] = self.map['treasure']
            else: info[1] = self.map['treasure']
            done = True
        
        next_obs = self.get_obs()

        return next_obs, reward, done, info

    def get_global_obs(self):
        return self.generate_map_obs()

    def get_agt_obs(self, i):
        obs = self.generate_map_obs()[self.agt_pos[i][0]-1:self.agt_pos[i][0]+2,self.agt_pos[i][1]-1:self.agt_pos[i][1]+2]
        return obs

    def get_all_agt_obs(self):
        return [self.get_agt_obs(i) for i in range(self.agent_num)]

    def get_partial_obs(self, i):
        partial_obs = self.occupancy[self.agt_pos[i][0]-1:self.agt_pos[i][0]+2,self.agt_pos[i][1]-1:self.agt_pos[i][1]+2].reshape(1,-1)
        rel_dis = np.array(self.agt_pos[i]) - np.array(self.treasure_pos)
        return np.concatenate([np.squeeze(partial_obs), self.agt_pos[1-i], rel_dis])

    def get_obs(self):
        return [self.get_partial_obs(i) for i in range(self.agent_num)]

    def get_state(self):
        return np.concatenate([line_state for line_state in self.occupancy], axis = 0)


    def plot_scene(self, idx):
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        plt.xticks([])
        plt.yticks([])
        ax2 = fig.add_subplot(gs[2, 0:1])
        plt.xticks([])
        plt.yticks([])
        ax3 = fig.add_subplot(gs[2, 1:2])
        plt.xticks([])
        plt.yticks([])

        ax1.imshow(self.get_global_obs())
        ax2.imshow(self.get_agt_obs(0))
        ax3.imshow(self.get_agt_obs(1))
        plt.savefig('./envs/alice-and-bob/images/step_{}'.format(idx))
        plt.clf()

    def render(self):

        obs = self.get_global_obs()
        enlarge = 30

        new_obs = np.zeros((self.length*enlarge, self.width*enlarge, 3))
        for i in range(self.length):
            for j in range(self.width):
                if np.sum(obs[i][j]) > 0:
                    cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), obs[i][j]*255, -1)

        cv2.imshow('image', new_obs)
        cv2.waitKey(100)
