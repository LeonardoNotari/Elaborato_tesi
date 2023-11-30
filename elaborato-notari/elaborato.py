import pickle
import gzip
import open_clip
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np

#%% CARICAMENTO LISTE


labels = ["sadness", "awe", "amusement", "contentment", "excitement", "fear", "disgust", "anger"]
X_images = []
X_utterances = []
Y_images = []
lbl_emotions_img = []

X_texts = []
Y_texts = []
lbl_emotions_txt = []

file_path = 'elaborato-notari/dataset_testi/index_images.pkl'
with open(file_path, 'rb') as file:
    index_images = pickle.load(file)
X_images = index_images

file_path = 'elaborato-notari/dataset_testi/utterances.pkl'
with open(file_path, 'rb') as file:
    X_utterances = pickle.load(file)

file_path = 'elaborato-notari/dataset_testi/emotions.pkl'
with open(file_path, 'rb') as file:
    Y_images = pickle.load(file)

for i in Y_images:
    lbl_emotions_img.append(labels.index(i))


file_path = 'elaborato-notari/dataset_testi/texts_map_zip.pkl.gz'
with gzip.open(file_path, 'rb') as file:
    texts_map = pickle.load(file)

for i in texts_map:
    X_texts.append(i[2])
    Y_texts.append(i[3])

for i in Y_texts:
    lbl_emotions_txt.append(labels.index(i))

#%% DISTRIBUZIONE UNIFORME

X_texts_tr = []
Y_texts_tr = []
lbl_emotions_txt_tr = []
X_texts_te = []
Y_texts_te = []
lbl_emotions_txt_te = []
for i in range(len(X_texts)):
  if lbl_emotions_txt_tr.count(lbl_emotions_txt[i]) < 6250:
    X_texts_tr.append(X_texts[i])
    Y_texts_tr.append(Y_texts[i])
    lbl_emotions_txt_tr.append(lbl_emotions_txt[i])
  else:
    if lbl_emotions_txt_te.count(lbl_emotions_txt[i]) < 250:
      X_texts_te.append(X_texts[i])
      Y_texts_te.append(Y_texts[i])
      lbl_emotions_txt_te.append(lbl_emotions_txt[i])


#%% CLASSE DATALOADER TESTI

class texts_emotions_dataset():
    def __init__(self, texts, emotions, labels):
        self.title  = tokenizer(texts)
        self.emotions  = tokenizer(emotions)
        self.labels = labels

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        title = self.title[idx]
        emotion = self.emotions[idx]
        label = self.labels[idx]
        return title, emotion, label

#%% MODELLO CLIP

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

#%% FINETUNING SUI TESTI

batch_size = 128
Xtr = X_texts_tr
Ytr = Y_texts_tr

lbl_tr = lbl_emotions_txt_tr
train_dataset = texts_emotions_dataset(Xtr, Ytr, lbl_tr)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.000005,eps=1e-6,weight_decay=0.01)
loss = nn.CrossEntropyLoss()

num_epochs = 4
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()
        texts, emotions, lbls = batch
        logits_per_texts = model.encode_text(texts)
        logits_per_emotions = model.encode_text(texts)
        ground_truth = torch.tensor(lbls, dtype=torch.long)
        total_loss = (loss(logits_per_texts, ground_truth) + loss(logits_per_emotions, ground_truth))/2
        total_loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

#%%

count = 0
count_bi = 0
X = X_texts_te
Y = Y_texts_te
lbl = labels
for i in range(len(Y)):
    text = tokenizer(X[i])
    with torch.no_grad(), torch.cuda.amp.autocast():
      emotions = tokenizer(lbl)
      emotions_features = model.encode_text(emotions)
      text_features = model.encode_text(text)
      emotions_features /= emotions_features.norm(dim=-1, keepdim=True)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      probs = (100.0 * text_features @ emotions_features.T).softmax(dim=-1)
    probs = probs.to('cpu')[0]
    index = np.where(probs == max(probs))
    if Y[i] == lbl[index[0][0]]:
        count += 1
    if index[0][0] in [0, 5, 6, 7] and lbl.index(Y[i]) in [0, 5, 6, 7]:
        count_bi += 1
    if index[0][0] in [1, 2, 3, 4] and lbl.index(Y[i]) in [1, 2, 3, 4]:
        count_bi += 1
print(count)
print(count_bi)

#%% DATALOADER PER IMMAGINI

class images_emotions_dataset():
    def __init__(self, images, emotions, labels):
        self.images = images
        self.emotions = tokenizer(emotions)
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        folder_path = "elaborato_notari/immagini_opere/"
        filename = str(self.images[idx]) + '.jpg'
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        image = preprocess(image)
        label = self.labels[idx]
        emotion = self.emotions[idx]
        return image, emotion, label

#%% FINETUNING SULLE IMMAGINI

batch_size = 64
k = 50
Xtr = X_images[:k]
Ytr = Y_images[:k]
lbl_images = lbl_emotions_img[:k]
train_dataset = images_emotions_dataset(Xtr, Ytr, lbl_images)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000003,eps=1e-6,weight_decay=0.001)
loss = nn.CrossEntropyLoss()

num_epochs = 1
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()
        images, emotions, lbl_images = batch
        outputs = model(images, emotions)
        logits_per_images = outputs[0]
        logits_per_emotions = outputs[1]
        ground_truth = torch.tensor(lbl_images, dtype=torch.long)
        total_loss = (loss(logits_per_images, ground_truth) + loss(logits_per_emotions, ground_truth))/2
        total_loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
        torch.cuda.empty_cache()

#%%

model.eval()
count = 0
count_bi = 0
#X = X_images[:1000]
#Y = Y_images[:1000]
l = len(X_images)
X = X_images[l-100:l]
Y = Y_images[l-100:l]
lbl = labels
for i in range(len(Y)):
    image = X[i]
    image = preprocess(image).unsqueeze(0)
    text = tokenizer(lbl)
    with torch.no_grad(), torch.cuda.amp.autocast():
      outputs = model(image, text)
      logits_per_image = outputs[0]
      logits_per_text = outputs[1]
      probs = (100.0 * logits_per_image @ logits_per_text.T).softmax(dim=-1)
      probs = probs.to('cpu')[0]
    index = np.where(probs == max(probs))
    if index[0][0] in [0, 5, 6, 7] and lbl.index(Y[i]) in [0, 5, 6, 7]:
        count_bi += 1
    if index[0][0] in [1, 2, 3, 4] and lbl.index(Y[i]) in [1, 2, 3, 4]:
        count_bi += 1
    if Y[i] == lbl[index[0][0]]:
        count += 1
print(count)
print(count_bi)

#%%

class images_utterances_dataset():
    def __init__(self, images, texts, labels):
        self.images = images
        self.title  = tokenizer(texts)
        self.labels = labels

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        folder_path = "/content/drive/My Drive/paintings_images/"
        filename = str(self.images[idx]) + '.jpg'
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        image = preprocess(image)
        title = self.title[idx]
        label = self.labels[idx]
        return image, title, label


#%%  MLP SUGLI EMBEDDINGS COMBINATI DI TESTO E IMMAGINI

input_dim = 1024
hidden_dim = 256
output_dim = len(labels)

mlp_model = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, output_dim)
)

k = 1000
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.0005, eps=1e-6, weight_decay=0.02)
loss = nn.CrossEntropyLoss()
Xtr = X_images[:k]
Xtr_texts = X_utterances[:k]
lbl_tr = lbl_emotions_img[:k]
batch_size = 32
dataset = images_utterances_dataset(Xtr, Xtr_texts, lbl_tr)
train_dataloader = DataLoader(dataset, batch_size=batch_size)
mlp_model.train()
model.train()
num_epochs = 10
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()
        images, texts, ys = batch
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        combined_features = torch.cat((text_features, image_features), dim=1)
        outputs = mlp_model(combined_features)
        ground_truth = torch.tensor(ys, dtype=torch.long)
        total_loss = loss(outputs, ground_truth)
        total_loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

#%%

X = X_images[1000:1500]
Y = Y_images[1000:1500]
Xtexts = X_utterances[1000:1500]
mlp_model.eval()
model.eval()
count = 0
count_bi = 0
for i in range(100):
    image = X[i]
    text = Xtexts[i]

    folder_path = "/content/drive/My Drive/paintings_images/"
    filename = str(X_images[i]) + '.jpg'
    image_path = os.path.join(folder_path, filename)
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)

    text = tokenizer(text)
    lbl = tokenizer(labels)
    with torch.no_grad(), torch.cuda.amp.autocast():
      image_features = model.encode_image(image)
      text_features = model.encode_text(text)
      combined_features = torch.cat((text_features, image_features), dim=1)
      outputs = mlp_model(combined_features)
      probs = (100.0 * outputs).softmax(dim=-1)
      probs = probs.to('cpu')
    probs = probs[0]
    index = np.where(probs == max(probs))
    print(labels[index[0][0]])
    if index[0][0] in [0, 5, 6, 7] and lbl.index(Y[i]) in [0, 5, 6, 7]:
        count_bi += 1
    if index[0][0] in [1, 2, 3, 4] and lbl.index(Y[i]) in [1, 2, 3, 4]:
        count_bi += 1
    if Y[i] == labels[index[0][0]]:
        count += 1
print(count)
print(count_bi)
