import yaml
import kagglehub
import os


def download_asvspoof2019_data():
    path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
    return path


def get_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def get_data():
    config = get_config()

    if ('dataset' not in config) or ('local_path' not in config['dataset']):
        return download_asvspoof2019_data()

    local_data_path = config['dataset']['local_path']
    if (local_data_path is not None) and os.path.exists(local_data_path):
        return local_data_path

    return download_asvspoof2019_data()
