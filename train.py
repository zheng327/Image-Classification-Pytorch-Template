import copy
import time
import torch.cuda
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.model import LeNet
import torch.nn as nn
import pandas as pd


class Args:
    def __init__(self):
        self.train_rat = 0.8
        self.val_rat = 0.2
        self.batch_size = 124
        self.learning_rate = 0.001
        self.epochs = 5
        self.seed = 42


args = Args()
torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def train_val_data_process():
    train_data = MNIST(root='./data/train',
                       train=True,
                       download=True,
                       transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))
    train_data, val_data = torch.utils.data.random_split(train_data,
                                                         [round(len(train_data)*args.train_rat),
                                                          round(len(train_data)*args.val_rat)])

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)

    val_loader = DataLoader(dataset=val_data,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(args.epochs):
        print("Epoch {}/{}".format(epoch+1, args.epochs))
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0
        for step, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()

            output = model(inputs)
            pred = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(pred == labels.data)
            train_num += inputs.size(0)

        for step, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.eval()
            output = model(inputs)
            pred = torch.argmax(output, dim=1)
            loss = criterion(output, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(pred == labels.data)
            val_num += inputs.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.item() / val_num)

        print("{} Train Loss: {:.4f} Train Acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} Val Loss: {:.4f} Val Acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            torch.save(model.state_dict(), 'best_model.pth')
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    torch.save(best_model_wts, "./best_model.pth")

    train_process = pd.DataFrame(data={"epoch": range(args.epochs),
                                       "train_loss": train_loss_all,
                                       "val_loss": val_loss_all,
                                       "train_acc": train_acc_all,
                                       "val_acc": val_acc_all})
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss, "ro-", label="Train Loss")
    plt.plot(train_process["epoch"], train_process.val_loss, "bs-", label="Val Loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc, "ro-", label="Train Acc")
    plt.plot(train_process["epoch"], train_process.val_acc, "bs-", label="Val Acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.show()


if __name__ == "__main__":
    LeNet = LeNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model(LeNet, train_dataloader, val_dataloader)
    matplot_acc_loss(train_process)
