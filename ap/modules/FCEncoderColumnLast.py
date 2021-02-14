import torch
from .FCEncoder import FCEncoder


class FCEncoderColumnLast(FCEncoder):

    def __init__(self, config, logger, column):

        super(FCEncoderColumnLast, self).__init__(config, logger)

        self.column = column

        # no gradients for column
        for parameter in self.column.parameters():
            parameter.requires_grad = False

        self.output_size += self.column.output_size

    def forward(self, x):

        x_input = x

        for i in range(len(self.neurons)):

            last = i == len(self.neurons) - 1

            x = self.fcs[i](x)

            if not last or self.activation_last:

                if self.use_norm:
                    x = self.norms[i](x)

                x = self.relu(x)

        # pass the input through the column network
        with torch.no_grad():
            y = self.column(x_input)
        # concatenate the final features
        return torch.cat([x, y], dim=1)
