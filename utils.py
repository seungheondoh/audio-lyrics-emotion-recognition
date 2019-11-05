import torch
import torch.nn as nn
import sys
import numpy as np

def set_device(model, device):
    if type(device) is int:
        if device > 0:
            torch.cuda.set_device(device - 1)
            model.cuda(device - 1)
            floatTensor = torch.cuda.FloatTensor  
            longTensor = torch.cuda.LongTensor
    elif type(device) is list:
        devices = [i - 1 for i in device]
        torch.cuda.set_device(devices[0])
        model = nn.DataParallel(model, device_ids=devices).cuda()
        floatTensor = torch.cuda.FloatTensor
        longTensor = torch.cuda.LongTensor
    return model, floatTensor, longTensor
