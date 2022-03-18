import torch


def masked_RMSE(prediction, y):
    diff = prediction - y
    n_observed = torch.count_nonzero(label)
    square_diff = torch.einsum('BH,BH->B', diff, diff)
    loss = torch.sqrt(torch.sum(square_diff)/ n_observed)
    return loss