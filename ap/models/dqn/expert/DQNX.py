from copy import deepcopy
import torch
import torch.nn.functional as F
from ....utils import torch as torch_utils


class DQNX:
    def __init__(self, fcn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.99,
                 num_primitives=1, sl=False, per=False):
        self.lr = lr
        self.fcn = fcn
        self.target_fcn = deepcopy(fcn)
        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.updateTarget()
        self.gamma = gamma
        self.criterion = torch_utils.WeightedHuberLoss()
        self.num_primitives = num_primitives
        self.action_space = action_space
        self.workspace = workspace
        self.heightmap_resolution = heightmap_resolution
        self.device = device
        self.sl = sl
        self.per = per

    def updateTarget(self):
        self.target_fcn.load_state_dict(self.fcn.state_dict())

    def forwardFCN(self, states, obs, target_net=False, to_cpu=False):
        fcn = self.fcn if not target_net else self.target_fcn
        obs = obs.to(self.device)
        q_value_maps = fcn(obs)[torch.arange(0, states.size(0)), states.long()]

        if to_cpu:
            q_value_maps = q_value_maps.cpu()
        return q_value_maps

    def getEGreedyActions(self, states, obs, eps, coef=0.01):
        with torch.no_grad():
            q_value_maps = self.forwardFCN(states, obs, to_cpu=True)
        q_value_maps += torch.randn_like(q_value_maps) * eps * coef
        pixels = torch_utils.argmax2d(q_value_maps).long()
        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        actions = torch.cat((x, y), dim=1)
        action_idx = pixels
        return q_value_maps, action_idx, actions

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        states = plan[:, 2:3]
        pixel_y = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_x = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()

        x = (pixel_y.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_x.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        actions = torch.cat((x, y), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y), dim=1)
        return action_idx, actions

    def update(self, batch):
        states, obs, pixel, rewards, next_states, next_obs, non_final_masks = self._loadBatchToDevice(batch)
        batch_size = obs.size(0)
        if self.sl:
            q = self.gamma ** rewards

        else:
            with torch.no_grad():
                q_map_prime = self.forwardFCN(next_states, next_obs, target_net=True)
                x_star = torch_utils.argmax2d(q_map_prime)
                q_prime = q_map_prime[torch.arange(0, batch_size), x_star[:, 0], x_star[:, 1]]
                q = rewards + self.gamma * q_prime * non_final_masks

        q_output = self.forwardFCN(states, obs)[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]
        loss = F.smooth_l1_loss(q_output, q)
        self.fcn_optimizer.zero_grad()
        loss.backward()
        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        td_error = torch.abs(q_output - q)
        return loss.item(), td_error


    def _loadBatchToDevice(self, batch):
        states = []
        images = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        dones = []
        step_lefts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs)
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs)
            dones.append(d.done)
            step_lefts.append(d.step_left)
        states_tensor = torch.stack(states).long().to(self.device)
        image_tensor = torch.stack(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        xy_tensor = torch.stack(xys).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        next_states_tensor = torch.stack(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        dones_tensor = torch.stack(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        step_lefts_tensor = torch.stack(step_lefts).to(self.device)

        return states_tensor, image_tensor, xy_tensor, rewards_tensor, next_states_tensor, next_obs_tensor, non_final_masks, step_lefts_tensor

    def _loadPrioritizedBatchToDevice(self, batch):
        batch, weights, idxes = batch
        weights = torch.from_numpy(weights).float().to(self.device)
        return self._loadBatchToDevice(batch), weights, idxes

    def loadModel(self, path_pre):
        fcn_path = path_pre + '_fcn.pt'
        print('loading {}'.format(fcn_path))
        self.fcn.load_state_dict(torch.load(fcn_path))
        self.updateTarget()

    def saveModel(self, path):
        torch.save(self.fcn.state_dict(), '{}_fcn.pt'.format(path))

    def getSaveState(self):
        return {
            'policy_net': self.fcn.state_dict(),
            'target_net': self.target_fcn.state_dict(),
            'optimizer': self.fcn_optimizer.state_dict()
        }

    def loadFromState(self, save_state):
        self.fcn.load_state_dict(save_state['policy_net'])
        self.target_fcn.load_state_dict(save_state['target_net'])
        self.fcn_optimizer.load_state_dict(save_state['optimizer'])

    def train(self):
        self.fcn.train()

    def eval(self):
        self.fcn.eval()

    def getModelStr(self):
        return str(self.fcn)
