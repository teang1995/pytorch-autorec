import torch


def masked_RMSE(prediction, y):
    diff = prediction - y
    n_observed = torch.count_nonzero(y)
    square_diff = torch.einsum('bh,bh->b', diff, diff)
    loss = torch.sqrt(torch.sum(square_diff)/ n_observed)
    return loss