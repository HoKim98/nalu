import os
import time
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('bmh')


class PlotManager:

    def __init__(self, dir_save='../plots'):
        self.dir_save = os.path.join(dir_save)
        if not os.path.exists(self.dir_save):
            os.mkdir(self.dir_save)
        self.dir_save = os.path.join(self.dir_save, time.strftime('%Y.%m.%d::%H:%M:%S'))
        os.mkdir(self.dir_save)

        self.data = ([[], [], []], [[], [], []])
        self.m = {0: 'Train', 1: 'Validate', 2: "Evaluate"}

    @classmethod
    def figure(cls):
        plt.clf()
        plt.title('Neural Arithmetic Logic Units')
        plt.xlabel('Epoch')
        plt.ylabel('MAE Loss')

    def add(self, y, x=None, z=0, step=False):
        x = len(self) if x is None else x
        for data, value in zip(self.data, (x, y)):
            data[z].append(value)
        if step:
            self.step()

    def step(self):
        self.figure()
        for i, (x, y) in enumerate(zip(self.data[0], self.data[1])):
            plt.plot(x, y, label=self.m[i])
        plt.legend()
        plt.savefig(os.path.join(self.dir_save, 'epoch_%d.svg' % len(self)))

    def __len__(self):
        return len(self.data[0][0])
