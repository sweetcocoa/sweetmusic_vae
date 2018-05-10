import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

"""
TODO :: 구글 논문 참고해서 모델 코드 완성 후 유닛테스트. 
"""

NUM_NOTES = 128
class MidiEncoder(nn.Module):
    def __init__(self, sequence_length, hidden_size):
        super(MidiEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=NUM_NOTES, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        """
        To get the end of the backward stream of RNN 
        https://github.com/pytorch/pytorch/issues/3587
        if bidirectional is true, then the contents of output should be like below: 
        [F_1, B_1]
        [F_2, B_2]
        [F_3, B_3] 
        and F_3 means the last output of forward, and B_1 means the last output of backward vectors.
        """
        forward_last = out[:, self.sequence_length - 1, :self.hidden_size]
        backward_last = out[:, 0, self.hidden_size:]

        return torch.cat([forward_last, backward_last], dim=1)


class MidiParameterize(nn.Module):
    def __init__(self, lstm_hidden_size, latent_dim):
        super(MidiParameterize, self).__init__()
        self.linear_mu = nn.Linear(lstm_hidden_size * 2, latent_dim)
        self.linear_sigma = nn.Linear(lstm_hidden_size * 2, latent_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        mu = self.linear_mu(x)
        sigma = self.softplus(self.linear_sigma(x))
        return mu, sigma

class MidiDecoder(nn.Module):

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    #
# class MidiVAE(nn.Module):
#     def __init__(self):
#         self.encoder = MidiEncoder()
#         self.mu_fcnet = nn.Linear()
#         self.sigma_fcnet = nn.Linear()
#         self.softplus = nn.Softplus()
#         self.conductor = MidiConductor()
#         self.decoder = MidiDecoder()
#
#     def forward(self, x):
#         ht_, _ht = self.encoder(x)  # 양 끝
#         ht = torch.cat([ht_, _ht], dim=1)
#         mu = self.mu_fcnet(ht)
#         sigma = self.softplus(self.sigma_fcnet(ht))
#         z = self.reparameterize(mu, sigma)
#         con = self.conductor(z)
#
#         pass
#
#     def decode(self, x):
#         pass
#
#     def reparameterize(self, x):
#         pass
