import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from kfp.v2.dsl import (
    component,
    Input,
    Artifact
)

from train_model_component.model import Net
from train_model_component.utils import train, evaluate

@component(
    packages_to_install=["pandas", "torch"],
    base_image="python:3.8",
    target_image="test_kubeflow_train_model"
)
def train_model(train_tensor_path: Input[Artifact], \
                     val_tensor_path: Input[Artifact], test_tensor_path: Input[Artifact]):
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    import torch.optim as optim
    
    from torch.utils.data import DataLoader

    train_tensor = torch.load(train_tensor_path.path)
    test_tensor = torch.load(test_tensor_path.path)
    val_tensor = torch.load(val_tensor_path.path)

    train_loader = DataLoader(train_tensor, batch_size=16, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=16, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=16, num_workers=2, shuffle=False)
    
    num_epoch = 1
    
    conv_model = Net()

    optimizer = optim.Adam(params=conv_model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if torch.cuda.is_available():
        conv_model = conv_model.cuda()
        criterion = criterion.cuda()
    
    
    for n in range(num_epoch):
        train(num_epoch, conv_model, exp_lr_scheduler, train_loader, optimizer, criterion)
        evaluate(val_loader, conv_model)
