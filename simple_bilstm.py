import pandas as pd
import random
import time
import torch
import torchtext as tt
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
SEED = 1234
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True


def load_data(path_to_csv: str, converted=False):
    LOGGER.info(f"[INFO] Loading data from {path_to_csv}")
    df = pd.read_csv(path_to_csv)
    if not converted:
        LOGGER.info(f"[INFO] Converting dataset to sentiment classification")
        df = df[['tweet_text', 'choose_one_category']]
        path_to_tmp_csv = str(path_to_csv) + ".tmp"
        LOGGER.info(f"[INFO] Saving binary classification dataset to {path_to_tmp_csv}")
        df.to_csv(path_to_tmp_csv, index=False)
    else:
        path_to_tmp_csv = path_to_csv

    LOGGER.info(f"[INFO] Loading data into tt.data.Dataset")
    dataset = tt.data.TabularDataset(path_to_tmp_csv,
                                     format="CSV",
                                     fields=[('text', TEXT),
                                             ('label', LABEL)],
                                     skip_header=True
                                     )

    LOGGER.info(f"[INFO] Spitting dataset into train/valid")
    train_data, valid_data = dataset.split(split_ratio=0.8)
    return train_data, valid_data


def load_pred_data(path_to_csv: str):
    LOGGER.info(f"[INFO] Loading data from {path_to_csv}")
    df = pd.read_csv(path_to_csv)
    path_to_tmp_csv = path_to_csv + ".tmp"
    df[['text']].to_csv(path_to_tmp_csv, index=False)

    LOGGER.info(f"[INFO] Loading data into tt.data.Dataset")
    dataset = tt.data.TabularDataset(path_to_tmp_csv,
                                     format="CSV",
                                     fields=[('text', PREDTEXT),],
                                     skip_header=True
                                     )

    return dataset


def build_vocabulary():
    LOGGER.info(f"[INFO] Building vocabulary for text and labels")
    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    LOGGER.info(f"[INFO] Vocabulary (text) size: {len(TEXT.vocab)}")
    LOGGER.info(f"[INFO] Most common tokens in vocabulary (text):\n\t {TEXT.vocab.freqs.most_common(10)}")
    LOGGER.info(f"[INFO] Vocabulary (labels):\n {str(LABEL.vocab.stoi)}")


def build_data_iterators():
    train_iterator, valid_iterator = tt.data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=DEVICE)

    return train_iterator, valid_iterator


class LSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=False,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = output_dim

    def forward(self, text, text_lengths):
        # text:(sent_len, batch_size), embedded:(sent_len, batch_size, emb_dim)
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)

        # output:(sent_len, batch_size ,hid_dim), hidden:(1,batch_size, hid_dim)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        hidden = self.dropout(hidden[-1,:,:])

        # logits:(batch_size, output_dim)
        logits = self.fc(hidden.squeeze(0))
        preds = F.softmax(logits, dim=1)
        return preds

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def multiclass_accuracy(preds: torch.Tensor, y: torch.Tensor, num_classes: int):
    """ Calculates accuracy for single-label multi-class case.

    :param preds: Model output class probabilities (batch_size x num_classes);
    :param y: Ground truth labels as indices (batch_size x 1)
    :return:
    """
    _, preds = torch.max(preds, 1)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(y, preds):
        confusion_matrix[t.long(), p.long()] += 1
    acc = confusion_matrix.diag().float() / confusion_matrix.sum(1)
    return tuple([a.item() if not torch.isnan(a) else 0 for a in acc])



def train(model, iterator, optimizer, criterion):
    epoch_loss = []
    epoch_acc = []

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(*batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = multiclass_accuracy(predictions, batch.label, model.num_labels)

        loss.backward()

        optimizer.step()

        epoch_loss += [loss.item()]
        epoch_acc += [acc]

    return np.mean(epoch_loss), np.mean(epoch_acc, axis=0)


def evaluate(model, iterator, criterion):
    epoch_loss = []
    epoch_acc = []

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(*batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = multiclass_accuracy(predictions, batch.label, model.num_labels)

            epoch_loss += [loss.item()]
            epoch_acc += [acc]

    return np.mean(epoch_loss), np.mean(epoch_acc, axis=0)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    PREDICT = True

    # hyperparameters
    DROPOUT = 0.2
    N_LAYERS = 2
    BIDIRECTIONAL = False

    # data
    MAX_VOCAB_SIZE = 25_000
    TEXT = tt.data.Field(tokenize='spacy', include_lengths=True)
    PREDTEXT = tt.data.Field(tokenize='spacy', include_lengths=True)
    LABEL = tt.data.LabelField(dtype=torch.long)
    train_data, valid_data = load_data(
        "data/CrisisNLP_labeled_data_crowdflower/pam_hagupit_pakflood_indflood_odile_caleq.csv",
        converted=False)
    build_vocabulary()

    # model
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    HIDDEN_DIM = 128
    OUTPUT_DIM = 9
    BATCH_SIZE = 64
    model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, PAD_IDX).to(DEVICE)
    print(f'The model has {model.count_parameters():,} trainable parameters')
    train_iterator, valid_iterator = build_data_iterators()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    if PREDICT:
        model.load_state_dict(torch.load("weather_content.pt"))
        pred_data = load_pred_data("data/nebraska_tweets.csv")
        PREDTEXT.vocab = TEXT.vocab
        pred_iterator = tt.data.BucketIterator(
            pred_data,
            batch_size=BATCH_SIZE,
            device=DEVICE)

        predictions = []
        model.eval()
        with torch.no_grad():
            for batch in pred_iterator:
                preds = model(*batch.text).squeeze(1)
                _, preds = torch.max(preds, 1)
                preds = [LABEL.vocab.itos[pred] for pred in preds]
                predictions += preds

        df = pd.read_csv("data/nebraska_tweets.csv")
        df['content_label'] = predictions
        df.to_csv("data/nebraska_tweets.csv")

    else:
        # train model
        N_EPOCHS = 150
        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'weather_content.pt')
                LOGGER.info("[INFO] saved model.")

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc}')
