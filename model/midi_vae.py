import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

"""
TODO :: 구글 논문 참고해서 모델 코드 완성 후 유닛테스트. 
"""

class MidiEncoder(nn.Module):
    def __init__(self):
        super(MidiEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        forward_last = out[:, :, 127]
        backward_last = out[:, :, 128]

        return torch.cat([forward_last, backward_last], dim=1)

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
