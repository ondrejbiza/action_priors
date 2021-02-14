import torch
import torch.nn as nn
from ..constants import Constants
from .FCEncoder import FCEncoder


class FCEncoderColumnSimple(nn.Module):

    def __init__(self, config, logger, column):

        super(FCEncoderColumnSimple, self).__init__()

        self.logger = logger
        self.column = column
        self.input_size = config[Constants.INPUT_SIZE]
        self.neurons = config[Constants.NEURONS]
        self.use_batch_norm = config[Constants.USE_BATCH_NORM]
        self.use_layer_norm = config[Constants.USE_LAYER_NORM]
        self.activation_last = config[Constants.ACTIVATION_LAST]

        # no gradients for column
        for parameter in self.column.parameters():
            parameter.requires_grad = False

        # the column should be an FC encoder with the same number of layers
        assert isinstance(self.column, FCEncoder)
        assert len(self.column.neurons) == len(self.neurons)

        assert not self.use_batch_norm or not self.use_layer_norm

        self.output_size = self.neurons[-1] + self.column.neurons[-1]
        self.use_norm = self.use_batch_norm or self.use_layer_norm
        self.relu = nn.ReLU(inplace=True)

        # initialize fully-connected layers
        self.fcs = nn.ModuleList(self.make_fcs_())

        # maybe initialize norm layers
        self.norms = None
        if self.use_batch_norm:
            self.norms = self.make_bns_()
        if self.use_layer_norm:
            self.norms = self.make_lns_()
        self.norms = nn.ModuleList(self.norms)

        # initialize variables
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if not self.use_norm:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is the input to the next layer of the main network
        # y is the input of the column network
        # both start the same
        y = x

        for i in range(len(self.neurons)):
            # true if last layer
            last = i == len(self.neurons) - 1

            # hidden features from the column network go into the main network
            # except for the first step
            if i > 0:
                x = torch.cat([x, y], dim=1)
            # step in the main network
            x = self.fcs[i](x)

            if not last or self.activation_last:

                if self.use_norm:
                    x = self.norms[i](x)

                x = self.relu(x)

            # step in the column network
            with torch.no_grad():
                y = self.column.fcs[i](y)

                if not last or self.column.activation_last:

                    if self.column.use_norm:
                        y = self.column.norms[i](y)

                    y = self.column.relu(y)
        # give the last column features to the prediction layer
        return torch.cat([x, y], dim=1)

    def make_fcs_(self):

        fcs = []

        for i in range(len(self.neurons)):
            if i == 0:
                fcs.append(nn.Linear(self.input_size, self.neurons[i], bias=not self.use_norm))
            else:
                # input size is the previous hidden layer size plus the column network hidden layer size
                fcs.append(
                    nn.Linear(self.neurons[i - 1] + self.column.neurons[i - 1],
                              self.neurons[i], bias=not self.use_norm)
                )

        return fcs

    def make_bns_(self):
        # TODO: not tested
        bns = []

        if self.activation_last:
            count = len(self.neurons)
        else:
            count = len(self.neurons) - 1

        for i in range(count):
            bns.append(nn.BatchNorm1d(self.neurons[i] + self.column.neurons[i]))

        return bns

    def make_lns_(self):
        # TODO: not tested
        lns = []

        if self.activation_last:
            count = len(self.neurons)
        else:
            count = len(self.neurons) - 1

        for i in range(count):
            lns.append(nn.LayerNorm(self.neurons[i] + self.column.neurons[i]))

        return lns
