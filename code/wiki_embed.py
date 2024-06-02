import pandas as pd
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaModel, RobertaTokenizer

# Load the model
model_name = "roberta-large"
model = RobertaModel.from_pretrained(model_name)

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_name)
df = pd.read_csv('../Data/wiki-labeled.csv')
gpt = df[df['label'] == 0]
human = df[df['label'] == 1]

def generate_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        #embeddings = torch.mean(outputs.last_hidden_state, dim=1)  # Mean pooling of token embeddings
        last_hidden_states = outputs.last_hidden_state
        embeddings = last_hidden_states[:,0,:]
    return embeddings.numpy()
import numpy as np
def remove_outliers(embeddings):
        # Calculate the mode of the embeddings
        mode_val = np.mean(embeddings[0], axis=None)
        
        # Replace outliers with the mode value
        embeddings[np.abs(embeddings - mode_val) > 2 * np.std(embeddings)] = mode_val
        
        return embeddings
def min_max_normalize(embeddings):
    # Find the minimum and maximum values in the embeddings
    min_val = np.min(embeddings)
    max_val = np.max(embeddings)
    
    # Normalize the embeddings to range [0, 255]
    normalized_embeddings = 255 * (embeddings - min_val) / (max_val - min_val)
    
    return normalized_embeddings.astype(np.uint8)  # Convert to uint8 for integer values between 0 and 255

def reshape_embeddings(embeddings):
    # Reshape the embeddings to 3D array
    return embeddings.reshape(32,32)
print("starting gpt embeddings ")
gpt['embeddings'] = gpt['text'].apply(generate_bert_embeddings)
gpt['normalized_embeddings'] = gpt['embeddings'].apply(min_max_normalize)
gpt['normalized_embeddings1'] = gpt['normalized_embeddings'].apply(remove_outliers)
gpt['reshaped_embeddings'] = gpt['normalized_embeddings1'].apply(reshape_embeddings)

print("starting human embeddings ")
human['embeddings'] = human['text'].apply(generate_bert_embeddings)
human['normalized_embeddings'] = human['embeddings'].apply(min_max_normalize)
human['normalized_embeddings1'] = human['normalized_embeddings'].apply(remove_outliers)
human['reshaped_embeddings'] = human['normalized_embeddings1'].apply(reshape_embeddings)

gpt.to_csv('../Data/gpt_wiki_embeddings1.csv', index=False)

human.to_csv('../Data/human_wiki_embeddings1.csv', index=False)