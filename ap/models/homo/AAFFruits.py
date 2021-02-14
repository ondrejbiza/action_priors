import torch
from torch import nn


class AAFFruits(nn.Module):

    def __init__(self, encoder, config, logger):

        super(AAFFruits, self).__init__()

        self.encoder = encoder
        self.logger = logger

        self.encoder_params = list(self.encoder.parameters())
        self.relu = nn.ReLU(inplace=False)
        self.decoder_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.validate_encoder_output_size_()
        self.build()

    def build(self):

        self.create_prob_prediction_vars_()

    def forward(self, states, actions):

        latent = self.relu(self.encoder([states, actions]))
        return self.fc_prob(latent)[:, 0]

    def get_accuracy(self, states, actions, labels):

        pred_probs = self.forward(states, actions)
        pred = (pred_probs > 0.0).type(torch.int32)
        eq = (pred == labels.type(torch.int32)).float()
        mean = torch.mean(eq)

        return mean

    def compute_loss(self, states, actions, labels):

        pred_probs = self.forward(states, actions)
        return self.compute_prob_loss_(labels, pred_probs)

    def compute_prob_loss_(self, labels, predicted_probs):

        loss = self.decoder_loss(predicted_probs, labels)
        return torch.mean(loss)

    def create_prob_prediction_vars_(self):

        self.fc_prob = nn.Linear(self.encoder.output_size, 1, bias=True)

        nn.init.kaiming_normal_(self.fc_prob.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.fc_prob.bias, 0)

    def validate_encoder_output_size_(self):

        assert isinstance(self.encoder.output_size, int)

    def save(self, path):

        torch.save(self.state_dict(), path)

    def load(self, path):

        state_dict = torch.load(path, map_location=self.fc_prob.weight.device)
        self.load_state_dict(state_dict)
