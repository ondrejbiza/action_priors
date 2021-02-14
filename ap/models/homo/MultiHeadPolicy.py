from torch import nn
from ...constants import Constants


class MultiHeadPolicy(nn.Module):

    def __init__(self, encoder, config, logger):

        super(MultiHeadPolicy, self).__init__()

        self.encoder = encoder
        self.logger = logger

        c = config
        self.num_heads = c[Constants.NUM_HEADS]
        self.num_actions = c[Constants.NUM_ACTIONS]
        self.tau = c[Constants.TAU]

        self.multi_head = None
        self.create_multi_head_()

        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, states, task_idx, log_softmax=False):

        latent = self.encoder(states)
        all_logits = self.multi_head(latent)
        logits = self.select_task_logits_(all_logits, task_idx)
        logits = logits / self.tau

        if log_softmax:
            return self.log_softmax(logits)
        else:
            return self.softmax(logits)

    def compute_loss(self, states, task_idx, teacher_policy):

        student_log_policy = self.forward(states, task_idx, log_softmax=True)
        return self.cross_entropy(teacher_policy, student_log_policy).mean(dim=0)

    def cross_entropy(self, teacher_policy, student_log_policy):

        return (- teacher_policy * student_log_policy).sum(dim=1)

    def create_multi_head_(self):

        self.multi_head = nn.Linear(self.encoder.output_size, self.num_heads * self.num_actions, bias=True)

    def select_task_logits_(self, all_logits, task_idx):

        if isinstance(task_idx, int):

            from_idx = self.num_actions * task_idx
            to_idx = self.num_actions * (task_idx + 1)

            return all_logits[:, from_idx: to_idx]

        else:

            assert task_idx.shape[0] == all_logits.shape[0]

            all_logits = all_logits.reshape((all_logits.size(0), self.num_heads, self.num_actions))
            return all_logits[list(range(all_logits.size(0))), task_idx]
