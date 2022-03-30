import torch
import torch.nn.functional as F

def old_old_masked_RMSE(prediction, y):
    diff = prediction - y
    n_observed = torch.count_nonzero(y)
    square_diff = torch.einsum('bh,bh->b', diff, diff)
    loss = torch.sqrt(torch.sum(square_diff)/ n_observed)
    return loss

def old_masked_RMSE(prediction, label):
    rating_counts = torch.count_nonzero(label,axis=1)
    mask = label != 0
    label = label * mask
    prediction = prediction * mask
    diff = prediction - label
    row_sum = torch.sum(diff * diff, axis=1)
    rvs = (1 / rating_counts) + 1e-6
    e = rvs * row_sum
    se = torch.sqrt(e)
    return torch.sum(se)


def masked_RMSE(prediction, label):
    rating_counts = torch.count_nonzero(label,axis=1)
    mask = label != 0
    label = label * mask
    prediction = prediction * mask
    loss = F.mse_loss(prediction, label)
    return loss