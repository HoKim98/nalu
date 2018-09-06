import torch
import torch.optim as optim

from utils.plot_manager import PlotManager


class TrainManager:

    def __init__(self, model, optimizer, data_manager, reduce_lr, *data_group):
        self.epoch = 0
        self.model = model
        self.op = optimizer
        self.lr_drop = optim.lr_scheduler.ReduceLROnPlateau(self.op) if reduce_lr else None
        #self.lr_drop = optim.lr_scheduler.ReduceLROnPlateau(self.op, min_lr=1e-3) if reduce_lr else None

        self.data_manager = data_manager
        self.data_train, self.data_val, self.data_eval = data_group

        self.plot_manager = PlotManager()

    def train(self, batch_size=1, verbose=False):
        self.epoch += 1
        N = len(self.data_train[0])
        loss_sum, accuracy = 0., 0
        t, loss = 0, 0.
        for _ in range(batch_size):
            for x, y in zip(*self.data_train):
                x = torch.LongTensor(x)
                x = x.unsqueeze(1)
                y_ = self.model(x)

                loss = loss + (y_ - y).abs().mean()
                t += 1
                if t == batch_size:
                    loss = loss.mean()
                    if not torch.isnan(y_):
                        self.op.zero_grad()
                        loss.backward()
                        self.op.step()
                    loss_sum += float(loss)
                    t, loss = 0, 0.

                if not torch.isnan(y_) and y_ <= 1e+9:
                    accuracy += int(int(y_ + .5) == int(y))
                else:
                    N -= 1

            self.data_train = self.data_manager.shuffle(self.data_train)
        loss_sum, accuracy = loss_sum / (N * batch_size), accuracy * 100 / (N * batch_size)
        if verbose:
            print('Train MAE ', self.epoch, loss_sum, accuracy)
        self.plot_manager.add(x=self.epoch, y=loss_sum, z=0)

    def validate(self, verbose=False):
        N = len(self.data_val[0])
        loss_sum, accuracy = 0., 0
        for x, y in zip(*self.data_val):
            x = torch.LongTensor(x)
            x = x.unsqueeze(1)
            y_ = self.model(x)

            loss = (y_ - y).abs().mean()

            loss_sum += float(loss)
            if not torch.isnan(y_) and y_ <= 1e+9:
                accuracy += int(int(y_ + .5) == int(y))
            else:
                N -= 1

        loss_sum, accuracy = loss_sum / N, accuracy * 100 / N
        if verbose:
            print('Valid MAE ', self.epoch, loss_sum, accuracy)
        if self.lr_drop is not None:
            self.lr_drop.step(loss_sum, self.epoch)
        self.plot_manager.add(x=self.epoch, y=loss_sum, z=1)

    def eval(self, verbose=True):
        N = len(self.data_eval[0])
        loss_sum, accuracy = 0., 0
        for x, y in zip(*self.data_eval):
            x = torch.LongTensor(x)
            x = x.unsqueeze(1)
            y_ = self.model(x)

            loss = (y_ - y).abs().mean()

            loss_sum += float(loss)
            if not torch.isnan(y_) and y_ <= 1e+9:
                accuracy += int(int(y_ + .5) == int(y))
            else:
                N -= 1

        loss_sum, accuracy = loss_sum / N, accuracy * 100 / N
        if verbose:
            print('Eval  MAE ', self.epoch, loss_sum, accuracy)
        self.plot_manager.add(x=self.epoch, y=loss_sum, z=2, step=True)
        return loss_sum if loss_sum == loss_sum else 1e+9
