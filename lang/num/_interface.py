class NumInterface:

    def __init__(self):
        pass

    def generate(self, nums):
        raise NotImplementedError

    def to_string(self, nums):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
