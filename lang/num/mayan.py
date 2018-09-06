from lang.num._interface import NumInterface


class Num(NumInterface):

    def __init__(self):
        super(NumInterface, self).__init__()
        self.m = {
            0: '',
            1: '-',  # not exists
            2: '@',
            3: '0',
            4: '1',
            5: ' ',
        }
        self.w = {v: k for k, v in self.m.items()}

    def generate(self, nums):
        def int_to_roman(num):
            if num == 0:
                return self.m[2]
            result = ''
            while num > 0:
                n = num % 20
                result += self.m[4] * (n // 5)
                result += self.m[3] * (n % 5)
                result += ' '
                num //= 20
            return result[-2::-1]
        return [[self.w[w] for w in int_to_roman(int(n))] for n in nums]

    def to_string(self, nums):
        return [''.join([self.m[x] for x in n]) for n in nums]

    def __len__(self):
        return len(self.m)
