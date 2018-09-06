import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from model.layer.nalu import NaluLayer


class Model(nn.Module):

    def __init__(self, num_embedding, num_hidden=32, num_lstm_layers=1, use_nalu=True, reduce_sum=True):
        super(Model, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_lstm_layers
        self.reduce_sum = reduce_sum

        self.embedder = nn.Embedding(num_embedding, num_hidden)
        self.lstm = nn.LSTM(num_hidden, num_hidden, num_lstm_layers, bias=True)
        self.final = NaluLayer(num_hidden, 1, 1, 0) if use_nalu else nn.Linear(num_hidden, 1)

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
        #x = x.mean(dim=0) if self.reduce_sum else x[-1]
        x = x.sum(dim=0) if self.reduce_sum else x[-1]
        x = self.final(x)
        return x


if __name__ == '__main__':

    import argparse

    import torch.optim as optim
    from utils.data_manager import DataManager
    from utils.train_manager import TrainManager

    parser = argparse.ArgumentParser(description='NALU - Language to Number Translation Tasks')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'ko', 'ja', 'roman', 'mayan'])
    parser.add_argument('--hidden', type=int, default=16, choices=[16, 32])
    parser.add_argument('--lstm-layers', type=int, default=1, choices=[1, 2])
    parser.add_argument('--use-nalu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--lr', type=float, default=1e-2, choices=[1e-2, 1e-3])
    parser.add_argument('--reduce-sum', type=int, default=1, choices=[0, 1])
    parser.add_argument('--reduce-lr', type=int, default=1, choices=[0, 1])
    parser.add_argument('--epochs', type=int, default=591)
    args = parser.parse_args()

    Num = __import__('lang.num.%s' % args.lang, fromlist=['Num'])
    data_manager = DataManager(getattr(Num, 'Num'))

    # TODO: Support more 'divide' option - now only English available
    data_train, data_val, data_eval = data_manager.divide(data_manager.generate())

    model = Model(len(data_manager), args.hidden, args.lstm_layers, args.use_nalu, args.reduce_sum)
    op = optim.Adam(model.parameters(), lr=args.lr)
    trainer = TrainManager(model, op, data_manager, args.reduce_lr, data_train, data_val, data_eval)

    loss_min = 1.e+9
    for i in range(args.epochs):
        trainer.train(verbose=True)
        trainer.validate()
        loss_min = min(loss_min, trainer.eval())
    print('min', loss_min)
