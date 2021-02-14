import numpy as np
import torch
import torch.nn.functional as F
from .DQNX import DQNX
from ....utils import torch as torch_utils


class DQNXRot(DQNX):
    def __init__(self, fcn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.99,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=True):
        self.heightmap_size = 90
        self.num_rotations = num_rotations
        self.half_rotation = half_rotation
        if self.half_rotation:
            self.rotations = torch.tensor([np.pi / self.num_rotations * i for i in range(self.num_rotations)])
        else:
            self.rotations = torch.tensor([(2*np.pi)/self.num_rotations * i for i in range(self.num_rotations)])

        super().__init__(fcn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives, sl, per)

    def getAffineMatrices(self, n, specific_rotations):
        if specific_rotations is None:
            rotations = [range(self.num_rotations) for _ in range(n)]
        else:
            rotations = specific_rotations
        affine_mats_before = []
        affine_mats_after = []
        for i in range(n):
            for rotate_idx in rotations[i]:
                if self.half_rotation:
                    rotate_theta = np.radians(rotate_idx * (180 / self.num_rotations))
                else:
                    rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                affine_mat_before = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)
                affine_mats_before.append(affine_mat_before)

                affine_mat_after = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                               [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)
                affine_mats_after.append(affine_mat_after)

        affine_mats_before = torch.cat(affine_mats_before)
        affine_mats_after = torch.cat(affine_mats_after)
        return affine_mats_before, affine_mats_after

    def forwardFCN(self, states, obs, target_net=False, to_cpu=False, specific_rotations=None):
        fcn = self.fcn if not target_net else self.target_fcn
        if specific_rotations is None:
            rotations = [range(self.num_rotations) for _ in range(obs.size(0))]
        else:
            rotations = specific_rotations
        diag_length = float(obs.size(2)) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - obs.size(2)) / 2)
        obs = obs.to(self.device)
        # pad obs
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        # expand obs into shape (n*num_rot, c, h, w)
        obs = obs.unsqueeze(1).repeat(1, len(rotations[0]), 1, 1, 1)
        obs = obs.reshape(obs.size(0) * obs.size(1), obs.size(2), obs.size(3), obs.size(4))

        affine_mats_before, affine_mats_after = self.getAffineMatrices(states.size(0), specific_rotations)
        # rotate obs
        flow_grid_before = F.affine_grid(affine_mats_before, obs.size())
        rotated_obs = F.grid_sample(obs, flow_grid_before, mode='bilinear')
        # forward network
        conv_output = fcn(rotated_obs)
        # rotate output
        flow_grid_after = F.affine_grid(affine_mats_after, conv_output.size())
        unrotate_output = F.grid_sample(conv_output, flow_grid_after, mode='bilinear')

        rotation_output = unrotate_output.reshape(
            (states.shape[0], -1, unrotate_output.size(1), unrotate_output.size(2), unrotate_output.size(3)))
        rotation_output = rotation_output.permute(0, 2, 1, 3, 4)
        predictions = rotation_output[torch.arange(0, states.size(0)), states.long()]
        predictions = predictions[:, :, padding_width: -padding_width, padding_width: -padding_width]
        if to_cpu:
            predictions = predictions.cpu()
        return predictions

    def getEGreedyActions(self, states, obs, eps, coef=0.01):
        with torch.no_grad():
            q_value_maps = self.forwardFCN(states, obs, to_cpu=True)
        q_value_maps += torch.randn_like(q_value_maps) * eps * coef
        action_idx = torch_utils.argmax3d(q_value_maps).long()
        rot_idx = action_idx[:, 0:1]
        rot = self.rotations[rot_idx]
        pixels = action_idx[:, 1:]
        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixels, rot_idx), dim=1)
        return q_value_maps, action_idx, actions

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        rot = plan[:, 2:3]
        states = plan[:, 3:4]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size-1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size-1)
        rot_id = (rot.expand(-1, self.num_rotations) - self.rotations).abs().argmin(1).unsqueeze(1)

        x = (pixel_x.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_y.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        rot = self.rotations[rot_id]
        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y, rot_id), dim=1)
        return action_idx, actions

    def update(self, batch):
        states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts = self._loadBatchToDevice(batch)
        batch_size = states.size(0)
        if self.sl:
            q = self.gamma ** step_lefts

        else:
            with torch.no_grad():
                divide_factor = 2
                small_batch_size = int(batch_size / divide_factor)
                qs = []
                for i in range(divide_factor):
                    s_next_states = next_states[small_batch_size*i:small_batch_size*(i+1)]
                    s_next_obs = (next_obs[0][small_batch_size*i:small_batch_size*(i+1)], next_obs[1][small_batch_size*i:small_batch_size*(i+1)])
                    s_rewards = rewards[small_batch_size*i:small_batch_size*(i+1)]
                    s_non_final_masks = non_final_masks[small_batch_size*i:small_batch_size*(i+1)]
                    q_map_prime = self.forwardFCN(s_next_states, s_next_obs, target_net=True)
                    q_prime = q_map_prime.reshape((small_batch_size, -1)).max(1)[0]
                    q = s_rewards + self.gamma * q_prime * s_non_final_masks
                    qs.append(q.detach())
                q = torch.cat(qs)

        q_output = self.forwardFCN(states, obs, specific_rotations=action_idx[:, 2:3].cpu())[torch.arange(0, batch_size), 0, action_idx[:, 0], action_idx[:, 1]]
        loss = F.smooth_l1_loss(q_output, q)
        self.fcn_optimizer.zero_grad()
        loss.backward()
        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        td_error = torch.abs(q_output - q).detach()
        return loss.item(), td_error
