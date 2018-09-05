import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform_

from model.layer.nalu import NaluLayer
from utils.data_manager import DataManager


from lang.num.roman import Num
#from lang.num.en import Num
#from lang.num.ko_kr import Num
data_util = DataManager(Num)


class Model(nn.Module):

    def __init__(self, num_hidden: int = 32, num_lstm_layers: int = 1):
        super(Model, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_lstm_layers

        self.embedder = nn.Embedding(len(data_util), num_hidden)
        self.lstm = nn.LSTM(num_hidden, num_hidden, num_lstm_layers)

        #self.final = nn.Linear(num_hidden, 1)
        self.final = NaluLayer(num_hidden, 1, 1, 0)

        self.hidden = None
        with torch.no_grad():
            for w in self.lstm.parameters():
                if w.dim() == 2:
                    xavier_uniform_(w)
                else:
                    w.fill_(0)
            xavier_uniform_(self.embedder.weight)

    def init_hidden(self, batch_size=1):
        self.hidden = (torch.zeros(self.num_layers, batch_size, self.num_hidden),
                       torch.zeros(self.num_layers, batch_size, self.num_hidden))

    def forward(self, x):
        self.init_hidden(1)
        x = self.embedder(x)
        x, self.hidden = self.lstm(x, self.hidden)
        #x = x.mean(dim=0)
        x = x[-1]
        x = self.final(x)
        return x


data_train, data_val, data_eval = data_util.divide(data_util.generate())


model = Model()
op = optim.Adam(model.parameters())
lr_drop = optim.lr_scheduler.ReduceLROnPlateau(op)


def train(i):
    global data_train
    N = len(data_train[0])
    loss_sum, accuracy = 0., 0
    for x, y in zip(*data_train):
        x = torch.LongTensor(x)
        x = x.unsqueeze(1)
        y_ = model(x)

        loss = (y_ - y).pow(2).mean()
        if not torch.isnan(y_):
            op.zero_grad()
            loss.backward()
            op.step()

        loss_sum += float(loss)
        if not torch.isnan(y_):
            accuracy += int(int(y_ + .5) == int(y))

    loss_sum, accuracy = loss_sum / N, accuracy * 100 / N
    #print('Train MSE ', i, loss_sum, accuracy)
    data_train = data_util.shuffle(data_train)


def validate(i):
    N = len(data_val[0])
    loss_sum, accuracy = 0., 0
    for x, y in zip(*data_val):
        x = torch.LongTensor(x)
        x = x.unsqueeze(1)
        y_ = model(x)

        loss = (y_ - y).abs().mean()

        loss_sum += float(loss)
        if not torch.isnan(y_):
            accuracy += int(int(y_ + .5) == int(y))

    loss_sum, accuracy = loss_sum / N, accuracy * 100 / N
    #print('Valid MAE ', i, loss_sum, accuracy)
    lr_drop.step(loss_sum, i)


def eval(i):
    N = len(data_eval[0])
    loss_sum, accuracy = 0., 0
    for x, y in zip(*data_eval):
        x = torch.LongTensor(x)
        x = x.unsqueeze(1)
        y_ = model(x)

        loss = (y_ - y).abs().mean()

        loss_sum += float(loss)
        if not torch.isnan(y_):
            accuracy += int(int(y_ + .5) == int(y))

    loss_sum, accuracy = loss_sum / N, accuracy * 100 / N
    print('Eval  MAE ', i, loss_sum, accuracy)
    global loss_min
    if loss_sum == loss_sum:
        loss_min = min(loss_min, loss_sum)


loss_min = 1.e+9
for i in range(3000):
    train(i)
    validate(i)
    eval(i)
print('min', loss_min)
