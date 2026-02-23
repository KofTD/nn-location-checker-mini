from pprint import pprint
from typing import override

import torch
import torch.nn as tnn
import torchvision  # pyright: ignore[reportMissingTypeStubs]
import torchvision.transforms.v2 as tt2  # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import DataLoader

from classifier import Classifier
from convolution import Convolution, TensorShape
from dataset import Dataset
from json_loader import ModuleLoader


class ConvolutionalNeuralNetwork(tnn.Module):
    def __init__(self, convolution: Convolution, classifier: Classifier):
        super().__init__()

        self.convolution = convolution.sequential()
        self.classifier = classifier.sequential()

    @override
    def forward(self, x: torch.Tensor):
        x = self.convolution(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    dataset = Dataset("./dataset/", tt2.Compose([tt2.Resize((227, 227))]))

    loader = DataLoader(
        dataset,
        64,
        True,
    )

    # print(len(loader))

    alexnet_convolution_loader = ModuleLoader(
        "./json_modules/alexnet_convolutions.json"
    )
    alexnet_classifier_loader = ModuleLoader("./json_modules/alexnet_classifier.json")

    alexnet_convolution = alexnet_convolution_loader.load(TensorShape(227, 227, 3))

    if not isinstance(alexnet_convolution, Convolution):
        raise RuntimeError("Wrong type")

    alexnet_classifier = alexnet_classifier_loader.load(alexnet_convolution.out_shape)

    if not isinstance(alexnet_classifier, Classifier):
        raise RuntimeError("Wrong type")

    # print(alexnet_convolution.out_shape)
    # print(alexnet_classifier.out_features)

    cnn_model = ConvolutionalNeuralNetwork(alexnet_convolution, alexnet_classifier)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn_model = cnn_model.to(device)

    learning_rate = 0.1
    num_epochs = 10

    loss_function = tnn.CrossEntropyLoss()
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


if __name__ == "__main__":
    main()
