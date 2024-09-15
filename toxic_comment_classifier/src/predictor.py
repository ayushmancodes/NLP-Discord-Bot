import numpy as np
import torch
from toxic_comment_classifier.src.model import LSTMToxicClassifier  # Updated import path


class Predictor:
    # Loading vocabulary
    vocab = np.load('toxic_comment_classifier/vocabulary/vocabulary.npy', allow_pickle=True).item()

    # Model parameter
    MAX_tokens = len(vocab)
    output_sequence_length = 119

    # Loading model
    model = LSTMToxicClassifier(MAX_tokens=MAX_tokens)
    model.load_state_dict(torch.load('toxic_comment_classifier/model_ckpt/model_weights.pth', map_location=torch.device('cpu'), weights_only=True))

    # Labels to predict
    labels =  ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def __init__(self):
        pass

    def _vectorize_sentence(sentence):
        vector = [Predictor.vocab.get(token, 0) for token in str(sentence).split()]  # Replace unseen words with 0
        diff = Predictor.output_sequence_length - len(vector) # for padding 
        if diff < 0:
            vector = vector[:Predictor.output_sequence_length]
        else:
            vector = vector + [0]*diff
        return vector
    
    def predict(self, text):
        input_vector = Predictor._vectorize_sentence(text)
        Predictor.model.eval()

        with torch.no_grad():
            output = Predictor.model(torch.tensor(input_vector))
            output = output.cpu().numpy()
            print(output)
        return {key:output[i] for i,key in enumerate(Predictor.labels)} 

