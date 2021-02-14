import torch.nn as nn
from ..constants import Constants


class FCEncoder(nn.Module):

    def __init__(self, config, logger):

        super(FCEncoder, self).__init__()

        self.logger = logger
        self.input_size = config[Constants.INPUT_SIZE]
        self.neurons = config[Constants.NEURONS]
        self.use_batch_norm = config[Constants.USE_BATCH_NORM]
        self.use_layer_norm = config[Constants.USE_LAYER_NORM]
        self.activation_last = config[Constants.ACTIVATION_LAST]

        assert not self.use_batch_norm or not self.use_layer_norm

        self.output_size = self.neurons[-1]
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

        for i in range(len(self.neurons)):

            last = i == len(self.neurons) - 1

            x = self.fcs[i](x)

            if not last or self.activation_last:

                if self.use_norm:
                    x = self.norms[i](x)

                x = self.relu(x)

        return x

    def make_fcs_(self):

        fcs = []

        for i in range(len(self.neurons)):
            if i == 0:
                fcs.append(nn.Linear(self.input_size, self.neurons[i], bias=not self.use_norm))
            else:
                fcs.append(nn.Linear(self.neurons[i - 1], self.neurons[i], bias=not self.use_norm))

        return fcs

    def make_bns_(self):

        bns = []

        if self.activation_last:
            count = len(self.neurons)
        else:
            count = len(self.neurons) - 1

        for i in range(count):
            bns.append(nn.BatchNorm1d(self.neurons[i]))

        return bns

    def make_lns_(self):

        lns = []

        if self.activation_last:
            count = len(self.neurons)
        else:
            count = len(self.neurons) - 1

        for i in range(count):
            lns.append(nn.LayerNorm(self.neurons[i]))

        return lns
