import torch
import torch.optim as optim


class TrainManager:

    def __init__(self, model, optimizer, data_manager, reduce_lr, *data_group):
        self.model = model
        self.op = optimizer
        self.lr_drop = optim.lr_scheduler.ReduceLROnPlateau(self.op) if reduce_lr else None

        self.data_manager = data_manager
        self.data_train, self.data_val, self.data_eval = data_group

        self.epoch = 0

    def train(self, verbose=False):
        self.epoch += 1
        N = len(self.data_train[0])
        loss_sum, accuracy = 0., 0
        for x, y in zip(*self.data_train):
            x = torch.LongTensor(x)
            x = x.unsqueeze(1)
            y_ = self.model(x)

            loss = (y_ - y).pow(2).mean()
            if not torch.isnan(y_):
                self.op.zero_grad()
                loss.backward()
                self.op.step()

            loss_sum += float(loss)
            if not torch.isnan(y_) and y_ <= 1e+9:
                accuracy += int(int(y_ + .5) == int(y))
            else:
                N -= 1

        loss_sum, accuracy = loss_sum / N, accuracy * 100 / N
        if verbose:
            print('Train MSE ', self.epoch, loss_sum, accuracy)
        self.data_train = self.data_manager.shuffle(self.data_train)

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
        return loss_sum if loss_sum == loss_sum else 1e+9
