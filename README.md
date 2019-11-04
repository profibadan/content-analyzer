# content-analyzer 
A simple LSTM built with pytorch to classify tweets into disaster-related categories.
Deprecated in favor of tweet-classifier library due to following limitations:  
- Doesn't use pre-trained word embeddings;  
- Builds vocabulary from training data with no pre-processing, leading to embeddings with poor generalizability;
