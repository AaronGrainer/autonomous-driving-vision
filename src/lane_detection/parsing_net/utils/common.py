import os

import torch


def save_model(net, optimizer, epoch, save_path):
    model_state_dict = net.state_dict()
    state = {"model": model_state_dict, "optimizer": optimizer.state_dict()}
    assert os.path.exists(save_path)
    model_path = os.path.join(save_path, f"ep{epoch}.pth")
    torch.save(state, model_path)
