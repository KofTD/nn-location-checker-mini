from src.data_loader import Data_loader
from src.dataset import Dataset


def main():
    dataset = Dataset("./dataset/")

    loader = Data_loader(dataset, 100, True)

    for image, marker in loader:
        print(marker)


if __name__ == "__main__":
    main()
