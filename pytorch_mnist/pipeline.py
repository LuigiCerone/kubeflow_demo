import kfp

from typing import NamedTuple
from kfp.v2 import dsl
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    OutputPath,
    InputPath,
    Artifact,
    Dataset,
    ClassificationMetrics,
    Metrics,
    HTML,
    Markdown
)
import argparse


download_link = 'https://github.com/kubeflow/examples/blob/master/digit-recognition-kaggle-competition/data/{file}.csv.zip?raw=true'


@component(
    packages_to_install=["wget", "pandas"],
    base_image="python:3.8"
)
def download_data(download_link: str, train: Output[Dataset], test: Output[Dataset]):
    import zipfile
    import wget
    import os
    import logging
    import pandas as pd

    data_path = '/tmp'

    # download files
    wget.download(download_link.format(file='train'),
                  f'{data_path}/train_csv.zip')
    wget.download(download_link.format(file='test'),
                  f'{data_path}/test_csv.zip')
    logging.info("Download completed.")

    with zipfile.ZipFile(f"{data_path}/train_csv.zip", "r") as zip_ref:
        zip_ref.extractall(data_path)

    with zipfile.ZipFile(f"{data_path}/test_csv.zip", "r") as zip_ref:
        zip_ref.extractall(data_path)

    logging.info('Extraction completed, path is %s', data_path)

    train_df = pd.read_csv(
        f"{data_path}/train.csv").to_csv(train.path, index=False)
    test_df = pd.read_csv(
        f"{data_path}/test.csv").to_csv(test.path, index=False)


@component(
    packages_to_install=["pandas", "scikit-learn", "torch"],
    base_image="python:3.8"
)
def pre_process_data(train: Input[Dataset], test: Input[Dataset], train_tensor_path: Output[Artifact], \
                     val_tensor_path: Output[Artifact], test_tensor_path: Output[Artifact]):
    import logging
    import torch
    import pandas as pd

    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset

    train_df = pd.read_csv(filepath_or_buffer=train.path)
    test_df = pd.read_csv(filepath_or_buffer=test.path)

    train_labels = train_df['label'].values
    train_images = (train_df.iloc[:, 1:].values).astype('float32')
    test_images = (test_df.iloc[:, :].values).astype('float32')

    # Training and Validation Split
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                                          stratify=train_labels, random_state=123,
                                                                          test_size=0.20)

    train_images = train_images.reshape(train_images.shape[0], 28, 28)
    val_images = val_images.reshape(val_images.shape[0], 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 28, 28)

    # train
    train_images_tensor = torch.tensor(train_images)/255.0
    train_labels_tensor = torch.tensor(train_labels)
    train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
    torch.save(train_tensor, train_tensor_path.path)

    # val
    val_images_tensor = torch.tensor(val_images)/255.0
    val_labels_tensor = torch.tensor(val_labels)
    val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)
    torch.save(val_tensor, val_tensor_path.path)

    # test
    test_tensor = torch.tensor(test_images)/255.0
    torch.save(test_tensor, test_tensor_path.path)


@component(
    packages_to_install=["pandas", "torch"],
    base_image="python:3.8"
)
def train_model(train_tensor_path: Input[Artifact], \
                     val_tensor_path: Input[Artifact], test_tensor_path: Input[Artifact]):
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    
    train_tensor = torch.load(train_tensor_path.path)
    test_tensor = torch.load(test_tensor_path.path)
    val_tensor = torch.load(val_tensor_path.path)

    train_loader = DataLoader(train_tensor, batch_size=16, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=16, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=16, num_workers=2, shuffle=False)
    
    num_epoch = 1


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
            self.conv_block = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2) 
            )
            
            self.linear_block = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(128*7*7, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, 10)
            )
            
        def forward(self, x):
            x = self.conv_block(x)
            x = x.view(x.size(0), -1)
            x = self.linear_block(x)
            
            return x
    conv_model = Net()

    optimizer = optim.Adam(params=conv_model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if torch.cuda.is_available():
        conv_model = conv_model.cuda()
        criterion = criterion.cuda()
    
    def train(num_epoch):
        conv_model.train()
        exp_lr_scheduler.step()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.unsqueeze(1)
            data, target = data, target
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                
            optimizer.zero_grad()
            output = conv_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1)% 100 == 0:
                pass
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    num_epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.data))
    
    def evaluate(data_loader):
        conv_model.eval()
        loss = 0
        correct = 0
        
        for data, target in data_loader:
            data = data.unsqueeze(1)
            data, target = data, target
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            
            output = conv_model(data)
            
            loss += F.cross_entropy(output, target, size_average=False).data

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
        loss /= len(data_loader.dataset)
            
        print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
            loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))
    

    for n in range(num_epoch):
        train(n)
        evaluate(val_loader)



@dsl.pipeline(name="digit-recognizer-pipeline",
              description="Performs Preprocessing, training and prediction of digits")
def digit_recognize_pipeline(download_link: str
                             ):

    # Create download container.
    generate_datasets = download_data(download_link)
    preprocess_tensors = pre_process_data(
        generate_datasets.outputs['train'], generate_datasets.outputs['test'])
    
    model = train_model(preprocess_tensors.outputs['train_tensor_path'], \
                        preprocess_tensors.outputs['val_tensor_path'], \
                        preprocess_tensors.outputs['test_tensor_path'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MNIST Kubeflow example")
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--no-run', dest='run', action='store_false')
    parser.set_defaults(run=True)

    args = parser.parse_args()

    # create client that would enable communication with the Pipelines API server
    client = kfp.Client()

    arguments = {"download_link": download_link}

    if args.run == 1:
        client.create_run_from_pipeline_func(digit_recognize_pipeline, arguments=arguments,
                                             experiment_name="mnist", mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE)
    else:
        kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
            pipeline_func=digit_recognize_pipeline, package_path='output_mnist.yaml')
        client.upload_pipeline_version(pipeline_package_path='output_mnist.yaml', pipeline_version_name="0.3",
                                       pipeline_name="MNIST example pipeline", description="Example pipeline")
