import numpy as np
import torch
from torch import nn


class SoftmaxClassifier(nn.Module):

    def __init__(self, encoder, num_classes):

        super(SoftmaxClassifier, self).__init__()

        self.encoder = encoder
        self.num_classes = num_classes

        self.create_fc_()
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        return self.fc(self.encoder(x))

    def get_prediction(self, x, logits=False, hard=False):

        pred = self.forward(x)

        if not logits:
            pred = self.softmax(pred)

        if hard:
            pred = torch.argmax(pred, dim=1)

        return pred

    def compute_loss_and_accuracy(self, x, y):

        pred = self.forward(x)
        loss = self.loss(input=pred, target=y)

        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        acc_np = np.mean(np.equal(np.argmax(pred_np, axis=1).astype(np.int32), y_np.astype(np.int32)).astype(np.float32))

        return torch.mean(loss), acc_np

    def create_fc_(self):

        self.fc = nn.Linear(self.encoder.output_size, self.num_classes, bias=True)

        nn.init.kaiming_normal_(self.fc.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.fc.bias, 0)
