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
load_data_path = "load"
preprocess_data_path = "preprocess"
model_path = "model"


@component(
    packages_to_install=["wget", "pandas"],
    base_image="python:3.8"
)
def download_data(download_link: str, train_df: Output[Dataset], test_df: Output[Dataset]):
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

    # Data Path
    train_data_path = data_path + '/train.csv'
    test_data_path = data_path + '/test.csv'

    # Loading dataset into pandas
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)


@component(
    packages_to_install=["pickle", "pandas"],
    base_image="python:3.8"
)
def demo(train_df: Input[Dataset], test_df: Input[Dataset]):
    import logging
    import os
    import pickle
    import pandas as pd

    logging.info("Called with %s", train_df.shape)


    return (print('Done!'))


# define pipeline
@dsl.pipeline(name="digit-recognizer-pipeline",
              description="Performs Preprocessing, training and prediction of digits")
def digit_recognize_pipeline(download_link: str,
                             data_path: str,
                             load_data_path: str,
                             preprocess_data_path: str,
                             model_path: str
                             ):

    # Create download container.
    download_container = download_data(download_link)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MNIST Kubeflow example")
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--no-run', dest='run', action='store_false')
    parser.set_defaults(run=True)

    args = parser.parse_args()

    # create client that would enable communication with the Pipelines API server
    client = kfp.Client()

    arguments = {"download_link": download_link,
                 "load_data_path": load_data_path,
                 "preprocess_data_path": preprocess_data_path,
                 "model_path": model_path}

    if args.run == 1:
        client.create_run_from_pipeline_func(digit_recognize_pipeline, arguments=arguments,
                                             experiment_name="mnist", mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE)
    else:
        kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
            pipeline_func=digit_recognize_pipeline, package_path='output_mnist.yaml')
        client.upload_pipeline_version(pipeline_package_path='output_mnist.yaml', pipeline_version_name="0.3",
                                       pipeline_name="MNIST example pipeline", description="Example pipeline")
