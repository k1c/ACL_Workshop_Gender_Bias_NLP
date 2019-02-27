import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

df = pd.read_csv('./test_dataset.txt', sep=',', header=None, squeeze=True)
# Tokenized input
text = "A person must reside continuously in the Territory for 20 years before she may apply for permanent residence."
tokenized_text = tokenizer.tokenize(df)

print(tokenized_text)
# strip line and tokenise

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
#segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)

# If you have a GPU, put everything on cuda
# tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor)
    #get repr here and store
# We have a hidden states for each of the 12 layers in model bert-base-uncased
