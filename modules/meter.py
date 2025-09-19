class Meter(object):
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0.0, 0.0, 0.0, 0


    def __call__(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        return self
