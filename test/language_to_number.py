import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform_

from model.layer.nalu import NaluLayer


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.embedder = nn.Embedding(21, 16)

        self.num_layers = 2
        self.lstm = nn.LSTM(16, 16, self.num_layers)

        #self.final = nn.Linear(16, 1)
        self.final = NaluLayer(16, 1, 1, 0)

        xavier_uniform_(self.embedder.weight)

    def init_hidden(self, batch_size=1):
        self.hidden = (torch.zeros(self.num_layers, batch_size, 16),
                       torch.zeros(self.num_layers, batch_size, 16))

    def forward(self, x):
        self.init_hidden(1)
        x = self.embedder(x)
        x, self.hidden = self.lstm(x, self.hidden)
        x = x[-1]
        x = self.final(x)
        return x


from lang.num.ko_kr import Num
generator = Num()
def make_dataset(num=1, MIN=0, MAX=1000):
    y = torch.randint(MIN, MAX, (num, ))
    x = generator.generate(y)
    return x, y


def shuffle(data):
    data_x, data_y = data
    rand = torch.randperm(data_y.size(0))
    print(data_x)
    data_x = data_x[rand]
    print(data_x)
    data_y = data_y[rand]
    return data_x, data_y


data_train = make_dataset(1000)
data_eval = make_dataset(100)


model = Model()
op = optim.Adam(model.parameters())


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
    print('Train', i, loss_sum, accuracy)


def eval(i):
    N = len(data_eval[0])
    loss_sum, accuracy = 0., 0
    for x, y in zip(*data_eval):
        x = torch.LongTensor(x)
        x = x.unsqueeze(1)
        y_ = model(x)

        loss = (y_ - y).pow(2).mean()

        loss_sum += float(loss)
        if not torch.isnan(y_):
            accuracy += int(int(y_ + .5) == int(y))

    loss_sum, accuracy = loss_sum / N, accuracy * 100 / N
    print('Eval ', i, loss_sum, accuracy)
    return loss_sum


loss_min = 1.e+9
for i in range(1000):
    train(i)
    loss = eval(i)
    if loss == loss:
        loss_min = min(loss_min, loss)
    data_train = make_dataset(1000)
    #data_train = shuffle(data_train)
print('min', loss_min)
