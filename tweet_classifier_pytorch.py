import numpy as np
import spacy

import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from keras.preprocessing.text import Tokenizer as kerasTokenizer
from keras.preprocessing.sequence import pad_sequences

# random seed
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True


def load_california_earthquake_data():
    df = pd.read_csv('data/CrisisNLP_labeled_data_crowdflower/2014_California_Earthquake/2014_california_eq.csv')

    labels = df['choose_one_category'].unique().tolist()
    df['label'] = pd.to_numeric(df.choose_one_category.astype("category", categories=labels).cat.codes, downcast='unsigned')
    label_encodings = {i:l for i,l in enumerate(labels)}

    df_train = df.iloc[:int(len(df)*0.8)]
    df_dev = df.iloc[int(len(df)*0.8):int(len(df)*0.9)]
    df_test = df.iloc[int(len(df)*0.9):]

    data = {}
    data['x_train'] = df_train.tweet_text.values
    data['x_valid'] = df_dev.tweet_text.values
    data['x_test'] = df_test.tweet_text.values
    data['y_train'] = df_train.label.values
    data['y_valid'] = df_dev.label.values
    data['y_test'] = df_test.label.values

    return data, label_encodings


class Model(nn.Module):
    def __init__(self, embedding_matrix, num_classes):
        super(Model, self).__init__()

        ## Embedding Layer, Add parameter
        self.embedding = nn.Embedding(len(embedding_matrix), len(embedding_matrix[0]))
        et = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding.weight = nn.Parameter(et)
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(300, 40)
        self.linear = nn.Linear(40, 16)
        self.out = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_lstm, _ = self.lstm(h_embedding)
        max_pool, _ = torch.max(h_lstm, 1)
        linear = self.relu(self.linear(max_pool))
        out = self.out(linear)
        return out


# load the data
data, label_encodings = load_california_earthquake_data()

# create tokenizer
all_texts = [v for k,V in data.items() if k[0]=="x" for v in V]
tokenizer = kerasTokenizer(num_words=1000)
tokenizer.fit_on_texts(all_texts)
word_index = tokenizer.word_index

# preprocess the data
for k,v in data.items():
    if k[0] == "x":
        data[k] = pad_sequences(tokenizer.texts_to_sequences(v), maxlen=50)

# create iterator object for train and valid objects
x_train = torch.tensor(data['x_train'], dtype=torch.long)
y_train = torch.tensor(data['y_train'], dtype=torch.long)
train = TensorDataset(x_train, y_train)
trainloader = DataLoader(train, batch_size=128)

x_valid = torch.tensor(data['x_valid'], dtype=torch.long)
y_valid = torch.tensor(data['y_valid'], dtype=torch.long)
valid = TensorDataset(x_valid, y_valid)
validloader = DataLoader(valid, batch_size=128)

# create embedding matrix
nlp = spacy.load('en_core_web_md')
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = nlp.vocab.get_vector(word)
    embedding_matrix[i] = embedding_vector

# create model
model = Model(embedding_matrix, len(label_encodings))

# define loss and optimizers
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the Model
num_epochs=50
for epoch in range(num_epochs):
    train_loss, valid_loss = [], []

    model.train()
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    model.eval()
    for data, target in validloader:
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())
    print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss))

# Evaluate model
total=0
correct=0
for data, labels in validloader:
    output = model(data)
    _, preds_tensor = torch.max(output, 1)
    predicted = np.squeeze(preds_tensor.numpy())

    total += labels.size(0)
    correct += (predicted == labels.numpy()).sum()
print('Accuracy of the network: %d %%' % (100 * correct / total))