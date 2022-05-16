import torch
from common.eval import *
import models.transform_layers as TL
def init_center_c( train_loader, net, eps=0.1,dataset=None):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    c = torch.zeros(128, device=device)
    hflip = TL.HorizontalFlipLayer().to(device)
    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _ = data
            inputs = inputs.to(device)
            if dataset =='mnist':
                inputs = torch.cat((inputs, inputs, inputs), 2)
                inputs = inputs.reshape(inputs.shape[0], 3, 28, 28)
            images1, images2 = hflip(inputs.repeat(2, 1, 1, 1)).chunk(2)
            images_pair = torch.cat([images1, images2], dim=0)
            images_pair = simclr_aug(images_pair)
            _, outputs_aux = net(images_pair, simclr=True, penultimate=True, shift=True)
            outputs=outputs_aux['simclr']
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c