from num2words import num2words
from lang.num._interface import NumInterface as _Base


class NumInterface(_Base):

    def __init__(self, m, lang, m_other=None):
        self.m = m
        self.m_other = m if m_other is None else m_other
        self.w = {v: k for k, v in m.items()}
        self.lang = lang

    def generate_pre(self, x):
        raise NotImplementedError
        #return x

    def to_string_pre(self, x):
        raise NotImplementedError
        #return ' '.join(x)

    def generate(self, nums):
        return [[self.w[w] for w in self.generate_pre(self.num2words(n))]
                for n in nums]

    def to_string(self, nums):
        return [self.to_string_pre([self.m_other[x] for x in n]) for n in nums]

    def num2words(self, num):
        return num2words(int(num), lang=self.lang)

    def __len__(self):
        return len(self.m)
