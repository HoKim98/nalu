from lang.num._interface import NumInterface as _Base


class Num(_Base):

    def __init__(self):
        super(Num, self).__init__()
        self.m = {
            0: '',
            1: '-',  # not exists
            2: '0',  # not exists
            3: 'I',
            4: 'V',
            5: 'X',
            6: 'L',
            7: 'C',
            8: 'D',
            9: 'M',
        }
        self.w = {v: k for k, v in self.m.items()}
        self.map = [
            (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
            (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
        ]

    # Source from https://stackoverflow.com/questions/28777219/basic-program-to-convert-integer-to-roman-numerals
    # by Aristide
    def generate(self, nums):
        def int_to_roman(num):
            assert 0 <= num < 4000
            if num == 0:
                return self.m[2]
            result = ''
            for (arabic, roman) in self.map:
                factor, num = divmod(num, arabic)
                result += roman * factor
            return result
        return [[self.w[w] for w in int_to_roman(int(n))] + [self.EOS] for n in nums]

    def to_string(self, nums):
        return [''.join([self.m[int(x)] for x in n]) for n in nums]

    def __len__(self):
        return len(self.m)
