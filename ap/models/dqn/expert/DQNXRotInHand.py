import numpy as np
import torch
import torch.nn.functional as F
from .DQNXRot import DQNXRot
from ....utils import torch as torch_utils


class DQNXRotInHand(DQNXRot):
    def __init__(self, fcn, action_space, workspace, heightmap_resolution, device, lr=1e-4, gamma=0.99,
                 num_primitives=1, sl=False, per=False, num_rotations=8, half_rotation=True, patch_size=24):
        super().__init__(fcn, action_space, workspace, heightmap_resolution, device, lr, gamma, num_primitives, sl, per,
                         num_rotations, half_rotation)
        self.patch_size = patch_size
        self.empty_in_hand = torch.zeros((1, 1, self.patch_size, self.patch_size))
        self.his = None

        self.expert_sl = False

    def initHis(self, num_processes):
        self.his = torch.zeros((num_processes, 1, self.patch_size, self.patch_size))

    def getImgPatch(self, obs, center_pixel):
        batch_size = obs.size(0)
        img_size = obs.size(2)
        transition = (center_pixel - obs.size(2) / 2).float().flip(1)
        transition_scaled = transition / obs.size(2) * 2

        affine_mat = torch.eye(2).unsqueeze(0).expand(batch_size, -1, -1).float()
        if obs.is_cuda:
            affine_mat = affine_mat.to(self.device)
        affine_mat = torch.cat((affine_mat, transition_scaled.unsqueeze(2).float()), dim=2)
        flow_grid = F.affine_grid(affine_mat, obs.size())
        transformed = F.grid_sample(obs, flow_grid, mode='nearest', padding_mode='zeros')
        patch = transformed[:, :,
                int(img_size / 2 - self.patch_size / 2):int(img_size / 2 + self.patch_size / 2),
                int(img_size / 2 - self.patch_size / 2):int(img_size / 2 + self.patch_size / 2)]
        return patch

    def getCurrentObs(self, in_hand, obs):
        obss = []
        for i, o in enumerate(obs):
            obss.append((o.squeeze(), in_hand[i].squeeze()))
        return obss

    def getNextObs(self, patch, rotation, states_, obs_, dones):
        in_hand_img_ = self.getInHandImage(patch, rotation).cpu()
        in_hand_img_[1 - states_.bool()] = self.empty_in_hand.clone()
        in_hand_img_[dones.bool()] = self.empty_in_hand.clone()
        obss_ = []
        for i, o in enumerate(obs_):
            obss_.append((o, in_hand_img_[i]))
        return obss_

    def updateHis(self, patch, rotation, states_, obs_, dones):
        in_hand_img_ = self.getInHandImage(patch, rotation).cpu()
        in_hand_img_[~states_.bool()] = self.empty_in_hand.clone()
        in_hand_img_[dones.bool()] = self.empty_in_hand.clone()
        self.his = in_hand_img_

    def encodeInHand(self, input_img, in_hand_img):
        if input_img.size(2) == in_hand_img.size(2):
            return torch.cat((input_img, in_hand_img), dim=1)
        else:
            resized_in_hand = F.interpolate(in_hand_img, size=(input_img.size(2), input_img.size(3)),
                                            mode='nearest')
        return torch.cat((input_img, resized_in_hand), dim=1)

    def getInHandImage(self, patch, rot):
        with torch.no_grad():
            patch = patch.to(self.device)
            diag_length = float(patch.size(2)) * np.sqrt(2)
            diag_length = np.ceil(diag_length / 32) * 32
            padding_width = int((diag_length - patch.size(2)) / 2)
            patch = F.pad(patch, (padding_width, padding_width, padding_width, padding_width),
                          mode='constant', value=0)
            affine_mats = []
            for i in range(patch.shape[0]):
                rotate_theta = rot[i].item()
                affine_mat = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                         [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat.shape = (2, 3, 1)
                affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float().to(self.device)
                affine_mats.append(affine_mat)

            affine_mats = torch.cat(affine_mats)
            flow_grid = F.affine_grid(affine_mats, patch.size())
            depth_heightmap_rotated = F.grid_sample(patch, flow_grid, mode='bilinear')
            depth_heightmap_rotated = depth_heightmap_rotated[:, :, padding_width: -padding_width,
                                      padding_width: -padding_width]
            return depth_heightmap_rotated

    def forwardFCN(self, states, obs, target_net=False, to_cpu=False, specific_rotations=None):
        obs, in_hand = obs
        fcn = self.fcn if not target_net else self.target_fcn
        if specific_rotations is None:
            rotations = [range(self.num_rotations) for _ in range(obs.size(0))]
        else:
            rotations = specific_rotations
        diag_length = float(obs.size(2)) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - obs.size(2)) / 2)
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        # pad obs
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        # expand obs into shape (n*num_rot, c, h, w)
        obs = obs.unsqueeze(1).repeat(1, len(rotations[0]), 1, 1, 1)
        in_hand = in_hand.unsqueeze(1).repeat(1, len(rotations[0]), 1, 1, 1)
        obs = obs.reshape(obs.size(0) * obs.size(1), obs.size(2), obs.size(3), obs.size(4))
        in_hand = in_hand.reshape(in_hand.size(0) * in_hand.size(1), in_hand.size(2), in_hand.size(3), in_hand.size(4))

        affine_mats_before, affine_mats_after = self.getAffineMatrices(states.size(0), specific_rotations)
        # rotate obs
        flow_grid_before = F.affine_grid(affine_mats_before, obs.size())
        rotated_obs = F.grid_sample(obs, flow_grid_before, mode='bilinear')
        # forward network
        conv_output = fcn(rotated_obs, in_hand)
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

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0.01, random_actions=False):
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        with torch.no_grad():
            q_value_maps = self.forwardFCN(states, (obs, in_hand), to_cpu=True)
        q_value_maps += torch.randn_like(q_value_maps) * eps * coef
        action_idx = torch_utils.argmax3d(q_value_maps).long()
        pixels = action_idx[:, 1:]
        rot_idx = action_idx[:, 0:1]

        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps
        for i, m in enumerate(rand_mask):
            if m:
                if random_actions:
                    pixel_candidates = torch.nonzero(obs[i, 0]>-100.0)
                else:
                    pixel_candidates = torch.nonzero(obs[i, 0]>0.01)
                rand_pixel = pixel_candidates[np.random.randint(pixel_candidates.size(0))]
                pixels[i] = rand_pixel

        rand_phi = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.num_rotations)
        rot_idx[rand_mask, 0] = rand_phi.long()

        rot = self.rotations[rot_idx]
        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixels, rot_idx), dim=1)
        return q_value_maps, action_idx, actions

    def getEGreedyActionsWithMask(self, states, in_hand, obs, eps, policy, coef=0.01):
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        with torch.no_grad():
            q_value_maps = self.forwardFCN(states, (obs, in_hand), to_cpu=True)
        q_value_maps += torch.randn_like(q_value_maps) * eps * coef
        action_idx = torch_utils.argmax3d(q_value_maps).long()
        pixels = action_idx[:, 1:]
        rot_idx = action_idx[:, 0:1]

        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps
        for i, m in enumerate(rand_mask):
            if m:
                shape = q_value_maps.shape
                action = policy.act(
                    [obs[i:i+1], in_hand[i:i+1], states[i: i+1].long()],
                    q_value_maps[:, 0, :, :].detach().cpu().numpy().reshape((shape[0], shape[2] * shape[3])),
                    0
                )
                action = [action // 90, action % 90]
                pixels[i] = torch.tensor(np.array(action, dtype=np.long), device=pixels[i].device)

        rand_phi = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.num_rotations)
        rot_idx[rand_mask, 0] = rand_phi.long()

        rot = self.rotations[rot_idx]
        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixels, rot_idx), dim=1)
        return q_value_maps, action_idx, actions

    def reshapeNextObs(self, next_obs, batch_size):
        next_obs = zip(*next_obs)
        next_obs, next_in_hand = next_obs
        next_obs = torch.stack(next_obs)
        next_in_hand = torch.stack(next_in_hand)
        next_obs = next_obs.reshape(batch_size, next_obs.shape[-3], next_obs.shape[-2], next_obs.shape[-1])
        next_in_hand = next_in_hand.reshape(batch_size, next_in_hand.shape[-3], next_in_hand.shape[-2], next_in_hand.shape[-1])
        return next_obs, next_in_hand

    def _loadBatchToDevice(self, batch):
        states = []
        images = []
        in_hands = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        next_in_hands = []
        dones = []
        step_lefts = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs[0])
            in_hands.append(d.obs[1])
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs[0])
            next_in_hands.append(d.next_obs[1])
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
        states_tensor = torch.stack(states).long().to(self.device)
        image_tensor = torch.stack(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        in_hand_tensor = torch.stack(in_hands).to(self.device)
        if len(in_hand_tensor.shape) == 3:
            in_hand_tensor = in_hand_tensor.unsqueeze(1)
        xy_tensor = torch.stack(xys).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        next_states_tensor = torch.stack(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        next_in_hands_tensor = torch.stack(next_in_hands).to(self.device)
        if len(next_in_hands_tensor.shape) == 3:
            next_in_hands_tensor = next_in_hands_tensor.unsqueeze(1)
        dones_tensor = torch.stack(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        step_lefts_tensor = torch.stack(step_lefts).to(self.device)
        is_experts_tensor = torch.stack(is_experts).bool().to(self.device)

        return states_tensor, (image_tensor, in_hand_tensor), xy_tensor, rewards_tensor, next_states_tensor, \
               (next_obs_tensor, next_in_hands_tensor), non_final_masks, step_lefts_tensor, is_experts_tensor

    def update(self, batch):
        states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadBatchToDevice(batch)
        batch_size = states.size(0)
        if self.sl:
            q = self.gamma ** step_lefts

        else:
            with torch.no_grad():
                q_map_prime = self.forwardFCN(next_states, next_obs, target_net=True)
                q_prime = q_map_prime.reshape((batch_size, -1)).max(1)[0]
                q = rewards + self.gamma * q_prime * non_final_masks
                q = q.detach()
            if self.expert_sl:
                q_target_sl = self.gamma ** step_lefts
                q[is_experts] = q_target_sl[is_experts]

        q_output = self.forwardFCN(states, obs, specific_rotations=action_idx[:, 2:3].cpu())[torch.arange(0, batch_size), 0, action_idx[:, 0], action_idx[:, 1]]
        loss = F.smooth_l1_loss(q_output, q)
        self.fcn_optimizer.zero_grad()
        loss.backward()
        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        td_error = torch.abs(q_output - q).detach()

        return loss.item(), td_error
