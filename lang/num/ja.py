from lang.num._interface_num2words import NumInterface as _Base


class Num(_Base):

    def __init__(self):
        m = {
            0: '',
            1: 'マイナス',
            2: '零',
            3: '一',
            4: '二',
            5: '三',
            6: '四',
            7: '五',
            8: '六',
            9: '七',
            10: '八',
            11: '九',
            12: '十',
            13: '百',
            14: '千',
            15: '万',
            16: '億',
            17: '兆',
            18: '京',
        }
        super().__init__(m, 'ja')

    def generate_pre(self, x):
        return [w.replace('/', self.m[1]) for w in x.replace(self.m[1], '/')]

    def to_string_pre(self, x):
        return ''.join(x)
