

class BeamSearchNode(object):
    def __init__(self, h_s, p_node, idx, log_p, l):
        self.h_s = h_s
        self.p_node = p_node
        self.idx = idx
        self.log_p = log_p
        self.l = l
    def eval(self, alpha=1.0):
        reward = 0
        return self.log_p / float(self.l - 1 + 1e-6) + alpha * reward
