import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2BinaryClassifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", input_dim=256, dropout=0.3):
        super(Wav2Vec2BinaryClassifier, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, input_dim),  # Reduce to input_dim features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),  # Single neuron for binary classification
            nn.Sigmoid()
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Get the mean embedding over time (global average pooling)
        pooled_output = hidden_states.mean(dim=1)

        return self.classifier(pooled_output)
