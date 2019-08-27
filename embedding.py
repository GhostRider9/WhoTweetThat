import torch
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import os
import re
import sys
import gc
from tqdm import tqdm

text = []
for file in os.listdir('/home/zlp/data/CS221SAT/data/Holmes_Training_Data/'):
    with open(os.path.join('/home/zlp/data/CS221SAT/data/Holmes_Training_Data', file), 'r',errors="ignore") as f:
        text.extend(f.read().splitlines())

text = [x.replace('*', '') for x in text]
text = [re.sub('[^ \fA-Za-z0-9_]', '', x) for x in text]
text = [x for x in text if x != '']
print(text[:10])

raw_text = []
for x in text:
    raw_text.extend(x.split(' '))
raw_text = [x for x in raw_text if x != '']

vocab = set(raw_text)
vocab_size = len(vocab)


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(len(inputs), -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return(log_probs)


CONTEXT_SIZE = 2
batch_size = 1024
#device = torch.device('cuda:0')
losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size, embedding_dim=100,
             context_size=CONTEXT_SIZE*2)
#model.to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1)


data_iter = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                        shuffle=False, num_workers=12)

for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in tqdm(data_iter):
        context_ids = []
        for i in range(len(context[0])):
            context_ids.append(make_context_vector([context[j][i] for j in range(len(context))], word_to_ix))
        context_ids = torch.stack(context_ids)
#        context_ids = context_ids.to(device)
        context_ids = torch.autograd.Variable(context_ids.cuda())
        model.zero_grad()
        log_probs = model(context_ids)
        label = make_context_vector(target, word_to_ix)
#        label = label.to(device)
        label = torch.autograd.Variable(label.cuda())
        loss = loss_function(log_probs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print('epoch %d loss %.4f' %(epoch, total_loss))
print(losses)

