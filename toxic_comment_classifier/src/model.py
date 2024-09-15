import torch.nn as nn

class LSTMToxicClassifier(nn.Module):
    def __init__(self, MAX_tokens, name='LSTM_toxic_classifier'):
        super(LSTMToxicClassifier, self).__init__()
        self.name = name
        self.embedding = nn.Embedding(num_embeddings = MAX_tokens+1, embedding_dim = 128, padding_idx=0)
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features= 64, out_features=16)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=16, out_features=6)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_tensors):
        embeddings = self.embedding(input_tensors)
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout1(lstm_out)
        last_element = lstm_out[-1,:]
        fc_out = self.fc1(last_element)
        fc_out = self.relu(fc_out)
        out = self.dropout2(fc_out)
        out = self.fc2(out)
        prob = self.sigmoid(out)
        return prob