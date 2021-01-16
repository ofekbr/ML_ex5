from gcommand_dataset import GCommandLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


class model(nn.Module):
    def __init__(self, n_input=1, n_channel=5, stride=3, n_output=30):
        super().__init__()
        # self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=3, stride=stride)
        # self.bn1 = nn.BatchNorm2d(n_channel)
        # self.pool1 = nn.MaxPool2d(4)
        # self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=3)
        # self.bn2 = nn.BatchNorm2d(n_channel)
        # self.pool2 = nn.MaxPool2d(4)
        # self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=3)
        # self.bn3 = nn.BatchNorm2d(2 * n_channel)
        # self.pool3 = nn.MaxPool2d(4)
        # self.conv4 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=3)
        # self.bn4 = nn.BatchNorm2d(2 * n_channel)
        # self.pool4 = nn.MaxPool2d(4)
        # self.fc1 = nn.Linear(2 * n_channel, n_output)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear4 = nn.Linear(3840, 1024)
        self.bn4 = nn.BatchNorm1d(1024)

        self.linear5 = nn.Linear(1024, 128)
        self.bn5 = nn.BatchNorm1d(128)

        self.linear6 = nn.Linear(128, 30)

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(self.bn1(x))
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = F.relu(self.bn2(x))
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = F.relu(self.bn3(x))
        # x = self.pool3(x)
        # x = self.conv4(x)
        # x = F.relu(self.bn4(x))
        # x = self.pool4(x)
        # x = F.avg_pool1d(x, x.shape[-1])
        # x = x.permute(0, 2, 1)
        # x = self.fc1(x)
        # return F.log_softmax(x, dim=2)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = self.linear4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.linear5(x)
        x = F.relu(self.bn5(x))

        x = self.linear6(x)
        # x = F.avg_pool1d(x, x.shape[-1])
        # x = x.permute(0, 2, 1)
        # x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        # data = data.to(device)
        # target = target.to(device)

        # apply transform and model on whole batch directly on device
        # data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            # noinspection PyTypeChecker
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        # record loss
        losses.append(loss.item())


def valid(model, epoch):
    model.eval()
    correct = 0
    for data, target in valid_loader:
        # data = data.to(device)
        # target = target.to(device)

        # apply transform and model on whole batch directly on device
        # data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

    # noinspection PyTypeChecker
    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(valid_loader.dataset)} ({100. * correct / len(valid_loader.dataset):.0f}%)\n")


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


train_loader = torch.utils.data.DataLoader(GCommandLoader('gcommands/train'), batch_size=64, shuffle=True,
                                           pin_memory=True)
valid_loader = torch.utils.data.DataLoader(GCommandLoader('gcommands/valid'), batch_size=256, shuffle=True,
                                           pin_memory=True)
# test_loader = torch.utils.data.DataLoader(GCommandLoader('gcommands/train'), batch_size=10, shuffle=False,
#                                           pin_memory=True)

model = model()

# available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

log_interval = 45
n_epoch = 5

# pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
# transform = transform.to(device)
for epoch in range(1, n_epoch + 1):
    train(model, epoch, log_interval)
    valid(model, epoch)
    scheduler.step()
