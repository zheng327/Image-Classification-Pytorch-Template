import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.model import LeNet


def test_data_process():
    test_data = MNIST(root='./data/test',
                      train=False,
                      download=True,
                      transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))

    test_loader = DataLoader(dataset=test_data,
                             batch_size=1,
                             shuffle=True,
                             num_workers=0)

    return test_loader


def infer_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    test_corrects = 0
    test_num = 0
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            model.eval()
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            test_corrects += torch.sum(pred == labels.data)
            test_num += images.size(0)
            print("真实类：{} ---- 预测类：{}".format(classes[labels], classes[pred]))


if __name__ == "__main__":
    LeNet = LeNet()
    LeNet.load_state_dict(torch.load("./best_model.pth"))
    test_dataloader = test_data_process()
    infer_model(LeNet, test_dataloader)
