import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dataset import FishDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
data_path = './Archive'
train_set_path = './Archive/train_val.txt'
batch_size = 16
num_epochs = 5
learning_rate = 1e-4
def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
#============================= data loader =======================
train_files = readlines(train_set_path)
print(len(train_files))
trainset = FishDataset(data_path, train_files, is_train = True)
train_loader = DataLoader(trainset, batch_size, True, 
        num_workers = 6, pin_memory=True, drop_last=True)

#=================================================================

#============================= network ===========================
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d( 32, 64, kernel_size = 3, stride = 2 )
        self.d1 = nn.Linear(64 * 32 * 32, 128)
        self.d2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = x.flatten(start_dim = 1)

        x = self.d1(x)
        x = F.relu(x)

        logits = self.d2(x)
        out = F.softmax(logits, dim=1)
        return out
#=====================================================================
model = MyModel().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#=====================================================================

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

for epoch in range(num_epochs):
    train_running_loss = 0
    train_acc = 0
    model.train()
    for i, (img, label) in tqdm(enumerate(train_loader)):
        img = img.cuda()
        label = label.cuda()
        out = model(img)
        loss = criterion(out, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(out, label, batch_size)
    
    model.eval()
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / i, train_acc/i))         
