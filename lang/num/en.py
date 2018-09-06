from lang.num._interface_num2words import NumInterface


class Num(NumInterface):

    def __init__(self):
        m = {
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
            #33: 'thousand',
            #34: 'million',
            #35: 'billion',
            #36: 'trillion',
            #37: 'quadrillion',
        }
        super().__init__(m, 'en')

    def generate_pre(self, x):
        return x.replace('-', ' ').replace(',', ' ,').split(' ')

    def to_string_pre(self, x):
        return ' '.join(x).replace(' ,', ',')
