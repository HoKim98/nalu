import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform_

from model.layer.nalu import NaluLayer


from lang.num.en_us import Num
#from lang.num.ko_kr import Num
generator = Num()
def make_dataset(num=1, MIN=0, MAX=1000):
    y = torch.randint(MIN, MAX, (num, ))
    x = generator.generate(y)
    return x, y


class Model(nn.Module):

    def __init__(self, num_hidden: int = 32, num_lstm_layers: int = 1):
        super(Model, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_lstm_layers

        self.embedder = nn.Embedding(len(generator), num_hidden)
        self.lstm = nn.LSTM(num_hidden, num_hidden, num_lstm_layers)

        #self.final = nn.Linear(num_hidden, 1)
        self.final = NaluLayer(num_hidden, 1, 1, 0)

        self.hidden = None
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


def shuffle(data):
    data_x, data_y = data
    rand = torch.randperm(data_y.size(0))
    print(data_x)
    data_x = data_x[rand]
    print(data_x)
    data_y = data_y[rand]
    return data_x, data_y


data_train = make_dataset(1000)
data_val = make_dataset(100)
data_eval = make_dataset(100)


model = Model()
op = optim.Adam(model.parameters())
lr_drop = optim.lr_scheduler.ReduceLROnPlateau(op)


def train(i):
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
    print('Train MSE ', i, loss_sum, accuracy)


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
    print('Valid MAE ', i, loss_sum, accuracy)
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
for i in range(300):
    train(i)
    validate(i)
    eval(i)
    data_train = make_dataset(1000)
    #data_train = shuffle(data_train)
print('min', loss_min)
