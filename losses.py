loss_list = ['mse']

def mse():
    print("mse")
    return None


def find_loss(loss):
    if loss == 'mse':
        return mse
    else:
        return None