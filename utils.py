import pandas as pd
import kagglehub
import matplotlib.pyplot as plt


def load_protocol(protocol_file, names, sep=' ', index_col=None):
    pd_protocol = pd.read_csv(protocol_file, sep=sep, names=names, usecols=[0, 1, 3, 4],
                                  index_col=index_col, skipinitialspace=True)
    return pd_protocol


def download_asvspoof2019_data():
    path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
    return path


def process_data_frame(unprocessed_df):
    mapping = {
      'bonafide': 0,
      'spoof': 1
    }
    df_processed = unprocessed_df[['file_name', 'label']]
    df_processed['label'] = df_processed['label'].map(mapping)

    print(df_processed.head())
    return df_processed


def plot_results(epoch_losses, epoch_accuracies, epochs):
    # Visualize loss and accuracy after training
    plt.figure(figsize=(12, 5))

    # Loss Graph
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', label='Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Accuracy Graph
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), epoch_accuracies, marker='o', color='green', label='Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()