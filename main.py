from pprint import pprint

import torch.nn
import torchvision.models

from src.chopper import Chopper
from src.data_loader import Data_loader
from src.dataset import Dataset, Marker
from src.segment import Segment


class ConvolutionalNeuralNetwork(torch.nn.Module):
    def __init__(self, head: Segment, tail: Segment):
        super().__init__()

        self.head = head.sequential()
        self.tail = tail.sequential()

    def forward(self, x):
        x = self.head(x)
        x = self.tail(x)
        return x


def main():
    dataset = Dataset("./dataset/")

    loader = Data_loader(dataset, 256, True)

    alexnet_chopper = Chopper(torchvision.models.alexnet())

    head = alexnet_chopper.head_chop(2)

    tail = Segment([torch.nn.Linear(6, 1536, True)])

    cnn_model = ConvolutionalNeuralNetwork(head, tail)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn_model = cnn_model.to(device)

    pprint(list(cnn_model.children()))

    learning_rate = 0.1
    num_epochs = 10

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate)

    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loader):
            images = images.requires_grad_().to(device)
            labels = labels.to(device)
            outputs = cnn_model(images)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch[{epoch}]: loss: {loss.item()}")


if __name__ == "__main__":
    main()
