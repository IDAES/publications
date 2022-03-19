from pyomo.common.timing import HierarchicalTimer as PyomoTimer


class TimingContext(object):

    def __init__(self, timer, name):
        self.timer = timer 
        self.name = name

    def __enter__(self):
        self.timer.start(self.name)
        return self

    def __exit__(self, ex_t, ex_v, ex_bt):
        self.timer.stop(self.name)


class HierarchicalTimer(PyomoTimer):

    def __call__(self, name):
        return TimingContext(self, name)

    def context(self, name):
        return TimingContext(self, name)
