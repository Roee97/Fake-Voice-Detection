import yaml
import kagglehub
import os
import pandas
import audio_utils


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


def process_data_frame(files_dir, unprocessed_df):
    mapping = {
      'bonafide': 0,
      'spoof': 1
    }
    df_processed = unprocessed_df[['file_name', 'label']]
    df_processed.loc[:, 'label'] = df_processed['label'].map(mapping)

    # Add info about augmentations
    augmented_files = [f for f in os.listdir(files_dir) if f.startswith("aug_")]

    # Create new rows for augmented files
    new_rows = []

    # for aug_file in augmented_files:
    #     # Remove "aug_" prefix and .flac suffix to get original file name
    #     original_file = aug_file[4:-5]
    #     matching_rows = df_processed[df_processed['file_name'] == original_file]
    #
    #     if not matching_rows.empty:
    #         # Duplicate row(s) and update the filename
    #         for _, row in matching_rows.iterrows():
    #             new_row = row.copy()
    #             new_row['file_name'] = aug_file[:-5]
    #             new_rows.append(new_row)

    # Append new rows to DataFrame
    if new_rows:
        print(len(new_rows))
        df_processed = pandas.concat([df_processed, pandas.DataFrame(new_rows)], ignore_index=True)

    print(df_processed.head())
    return df_processed
