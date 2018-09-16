import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from model.layer.nalu import NaluLayer


class Model(nn.Module):

    def __init__(self, num_embedding, num_hidden=32, num_lstm_layers=1, use_nalu=True, reduce_sum=True):
        super(Model, self).__init__()
        self.num_embedding = num_embedding
        self.num_hidden = num_hidden
        self.num_layers = num_lstm_layers
        self.reduce_sum = reduce_sum
        self.use_nalu = use_nalu

        self.embedder = nn.Embedding(num_embedding, num_hidden)
        self.encoder = None if use_nalu else nn.LSTM(num_hidden, num_hidden, num_lstm_layers)
        self.final = NaluLayer(num_hidden*2, 1, 1, 0) if use_nalu else nn.Linear(num_hidden, 1)

        if not use_nalu:
            with torch.no_grad():
                for w in self.encoder.parameters():
                    if w.dim() == 2:
                        xavier_uniform_(w)
                    else:
                        w.fill_(0)

    def init_hidden(self, num_hidden, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, num_hidden),
                torch.zeros(self.num_layers, batch_size, num_hidden))

    def one_hot(self, x):
        y = torch.eye(self.num_embedding)
        return y[x]

    def forward(self, x):
        return self.forward_nalu(x) if self.use_nalu else self.forward_lstm(x)

    def forward_lstm(self, x):
        hidden = self.init_hidden(self.num_hidden)
        y = self.embedder(x)
        y, hidden = self.encoder(y, hidden)
        y = y.sum(dim=0) if self.reduce_sum else y[-1]
        y = self.final(y)[0]
        return y

    def forward_nalu(self, x):
        y = self.embedder(x)
        t = torch.zeros(1)
        for i in range(0, y.size(0) - 1):
            t = t + self.final(y[i: i+2, 0].view(1, -1)).squeeze(1)
        return t


if __name__ == '__main__':

    import argparse

    import torch.optim as optim
    from utils.data_manager import DataManager
    from utils.train_manager import TrainManager

    parser = argparse.ArgumentParser(description='NALU - Language to Number Translation Tasks')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'ko', 'ja', 'roman', 'mayan'])
    parser.add_argument('--hidden', type=int, default=32, choices=[16, 32])
    parser.add_argument('--lstm-layers', type=int, default=1, choices=[1, 2])
    parser.add_argument('--use-nalu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--lr', type=float, default=1e-2, choices=[1e-2, 1e-3])
    parser.add_argument('--reduce-sum', type=int, default=0, choices=[0, 1])
    parser.add_argument('--reduce-lr', type=int, default=1, choices=[0, 1])
    parser.add_argument('--epochs', type=int, default=591)
    parser.add_argument('--batch-size', type=int, default=30)
    args = parser.parse_args()

    Num = __import__('lang.num.%s' % args.lang, fromlist=['Num'])
    data_manager = DataManager(getattr(Num, 'Num'))

    # TODO: Support more 'divide' option - now only English available
    data_train, data_val, data_eval = data_manager.divide(data_manager.generate(),
                                                          include=([*range(20), *range(20, 101, 10), 123,
                                                                    ], [], []))

    model = Model(len(data_manager), args.hidden, args.lstm_layers, args.use_nalu, args.reduce_sum)
    op = optim.Adam(model.parameters(), lr=args.lr)
    trainer = TrainManager(model, op, data_manager, args.reduce_lr, data_train, data_val, data_eval)

    loss_min = 1.e+9
    for i in range(args.epochs):
        trainer.train(args.batch_size, verbose=True)
        trainer.validate()
        loss_min = min(loss_min, trainer.eval())
    print('min', loss_min)
