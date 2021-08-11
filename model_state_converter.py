from collections import OrderedDict


def albert_for_pretraining(model_state):
    r_model_state = OrderedDict()
    for k, v in model_state.items():
        if k.startswith('albert.'):
            r_model_state[k[7:]] = v
    return r_model_state


def salbert_ft_model(model_state):
    r_model_state = OrderedDict()
    for k, v in model_state.items():
        if k.startswith('albert.'):
            r_model_state[k[7:]] = v
        if k.startswith('classifier'):
            pass
    return r_model_state
