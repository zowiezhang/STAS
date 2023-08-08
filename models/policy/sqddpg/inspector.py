import numpy as np



def inspector(args):
    assert args.replay is True
    assert args.q_func is True
    assert args.target is True
    assert args.gumbel_softmax is True
    assert args.epsilon_softmax is False
    assert args.online is True
    assert hasattr(args, 'sample_size')
