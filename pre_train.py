from torchnlp.datasets import imdb_dataset
from torchnlp.word_to_vector import GloVe
import torch.utils
import torchnlp.nn as tnn
import torch
import torchnlp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#参数
batchSize = 20
epoches = 10
seqLen = 100
vecDim = 300
num_fil = 32
LR = 0.001

#获取数据
train = imdb_dataset(train=True)
test = imdb_dataset(test=True)
vectors = GloVe()
num = train.__len__()

#把train test转换成词向量的形式
def word_vector(train):
    seq = []
    zeroTen = torch.zeros(1, vecDim)
    for i in range(num):
        # 先修改标签
        if train.__getitem__(i)['sentiment'] == 'pos':
            train.__getitem__(i)['sentiment'] = torch.tensor(0)
        else:
            train.__getitem__(i)['sentiment'] = torch.tensor(1)
        # 句子修改
        seq = train.__getitem__(i)['text'].split()
        if len(seq) > seqLen:
            x = torch.Tensor(vectors[seq[0]]).view(1, vecDim)
            for j in range(seqLen - 1):
                seq[j + 1] = vectors[seq[j + 1]].view(1, vecDim)
                x = torch.cat((x, seq[j + 1]), dim=0)
            train.__getitem__(i)['text'] = x
        else:
            k = len(seq)
            x = torch.Tensor(vectors[seq[0]]).view(1, vecDim)
            for j in range(k - 1):
                seq[j + 1] = vectors[seq[j + 1]].view(1, vecDim)
                x = torch.cat((x, seq[j + 1]), dim=0)

            for l in range(seqLen - k):
                x = torch.cat((x, zeroTen), dim=0)

            train.__getitem__(i)['text'] = x
    return train

train = word_vector(np.array(train))
test = word_vector(np.array(test))

#模型
class Attention(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = torch.bmm(attention_weights, context)

        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

#attention = Attention(vecDim)
#query = train['text']
#context = train['text']
#temp, weights = attention(query, context)

#train['text'] = temp


#批处理
trainLoader = torch.utils.data.DataLoader(train, batch_size=batchSize, shuffle=False)
testLoader = torch.utils.data.DataLoader(test, batch_size=batchSize, shuffle=False)


cnn = tnn.CNNEncoder(vecDim, num_fil, ngram_filter_sizes=[3])

#验证trainLoader里面的数据形状
# for i, data in enumerate(trainLoader, 0):
#     inputs = data['text']
#     print(inputs)
#     break

# 训练网络
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)


for epoch in range(epoches):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):

        inputs = data['text']
        labels = data['sentiment']
        optimizer.zero_grad()

        outputs = cnn(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

