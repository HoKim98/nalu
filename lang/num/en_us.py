from num2words import num2words


class Num:

    def __init__(self):
        self.m = {
            0: '',
            1: 'minus',
            2: 'zero',
            3: 'one',
            4: 'two',
            5: 'three',
            6: 'four',
            7: 'five',
            8: 'six',
            9: 'seven',
            10: 'eight',
            11: 'nine',
            12: 'ten',
            13: 'eleven',
            14: 'twelve',
            15: 'thirteen',
            16: 'fourteen',
            17: 'fifteen',
            18: 'sixteen',
            19: 'seventeen',
            20: 'eighteen',
            21: 'nineteen',
            22: 'twenty',
            23: 'thirty',
            24: 'forty',
            25: 'fifty',
            26: 'sixty',
            27: 'seventy',
            28: 'eighty',
            29: 'ninety',
            30: 'and',
            31: ',',
            32: 'hundred',
            33: 'thousand',
            34: 'million',
            35: 'billion',
            36: 'trillion',
            37: 'quadrillion',
        }
        self.w = {v: k for k, v in self.m.items()}

    def generate(self, nums):
        return [[self.w[w] for w in num2words(n).replace('-', ' ').replace(',', ' ,').split(' ')]
                for n in nums]

    def to_string(self, nums):
        return [' '.join([self.m[x] for x in n]).replace(' ,', ',') for n in nums]

    def __len__(self):
        return len(self.m)
