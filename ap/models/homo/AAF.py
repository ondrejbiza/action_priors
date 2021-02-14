import torch
from torch import nn
from ...constants import Constants


class AAF(nn.Module):

    def __init__(self, encoder, config, logger, build=True):

        super(AAF, self).__init__()

        self.encoder = encoder
        self.logger = logger

        c = config
        self.balance_loss = c[Constants.BALANCE_LOSS]

        self.relu = nn.ReLU(inplace=False)
        self.decoder_loss = nn.BCEWithLogitsLoss(reduction="none")

        if build:
            self.build()

    def build(self):

        self.create_pi_prediction_vars_()

    def forward(self, states):

        # |B|x|C|xHxW
        zs = self.relu(self.encoder([states, None], no_actions=True))
        # |B|x2xHxW
        qs = self.conv_pi(zs)

        hand_bits = states[2]

        # |B|xHxW
        qs = qs[list(range(len(hand_bits))), hand_bits]

        # |B|x(H*W)
        qs = qs.reshape((qs.size(0), qs.size(1) * qs.size(2)))

        return qs

    def compute_loss(self, states, labels, amb_labels=None):

        predictions = self.forward(states)

        assert predictions.shape == labels.shape

        loss = self.decoder_loss(predictions, labels)

        if self.balance_loss:
            assert amb_labels is None
            # TODO: implement this branch
            num_pixels = loss.size(1)
            num_zeros = torch.sum(labels == 0, dim=1)[:, None]
            num_ones = torch.sum(labels == 1, dim=1)[:, None]

            mask_zeros = (labels == 0).float()
            mask_ones = (labels == 1).float()

            weights = torch.zeros_like(loss)
            weights += mask_zeros * ((num_pixels / 2) / num_zeros)
            weights += mask_ones * ((num_pixels / 2) / num_ones)

            return torch.mean(torch.sum(loss * weights, dim=1), dim=0)
        else:
            if amb_labels is not None:
                mask = (1 - amb_labels).float()
                loss = loss * mask

            return torch.mean(torch.sum(loss, dim=1), dim=0)

    @torch.no_grad()
    def get_accuracy(self, states, labels):

        predictions = self.forward(states)
        predictions = (predictions >= 0.0).long()

        mask = (predictions == labels).float()

        acc0 = mask[labels == 0].mean()
        acc1 = mask[labels == 1].mean()
        acc = (acc0 + acc1) / 2

        return acc

    def create_pi_prediction_vars_(self):

        self.conv_pi = nn.Conv2d(
            32, 2, 1, stride=1, padding=0, bias=True
        )

        nn.init.kaiming_normal_(self.conv_pi.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.conv_pi.bias, 0)

    def save(self, path):

        torch.save(self.state_dict(), path)

    def load(self, path):

        state_dict = torch.load(path, map_location=self.conv_pi.weight.device)
        self.load_state_dict(state_dict)
