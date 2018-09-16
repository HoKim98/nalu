import torch
from random import shuffle
from lang.num.en import Num


class DataManager:

    def __init__(self, lang_model=Num):
        self._generator = lang_model()

    def generate(self, MIN=0, MAX=1000):
        y = torch.arange(MIN, MAX, dtype=torch.float)
        x = self._generator.generate(y)
        return x, y

    @classmethod
    def divide(cls, data, size=(169, 200, 631, ), include=([], [], [], )):
        result = [[[d[j] for j in inc] for d in data] for inc in include]
        indices = list(range(sum(size)))
        for idx in sorted([j for sub in include for j in sub], reverse=True):
            del indices[idx]
        shuffle(indices)
        size_before = 0
        for i, s in enumerate(size):
            s -= len(include[i])
            append = [[d[indices[idx]] for idx in range(size_before, s + size_before)] for d in data]
            result[i] = [r + a for r, a in zip(result[i], append)]
            size_before += s
        result = [[r[0], torch.Tensor(r[1])] for r in result]
        return result

    @classmethod
    def shuffle(cls, data):
        data_x, data_y = data
        rand = torch.randperm(data_y.size(0))
        data_x = [data_x[r] for r in rand]
        data_y = data_y[rand]
        return data_x, data_y

    def to_string(self, num):
        return self._generator.to_string([num])[0]

    def __len__(self):
        return len(self._generator)
