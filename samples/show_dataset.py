from random import sample

import torch
from matplotlib import pyplot as plot

from dataset import Dataset, Marker


def show_images(dataset_picks: list[tuple[torch.Tensor, int]]):
    num_showed_imgs_x = 5
    num_showed_imgs_y = 5

    figsize = (10, 10)
    fig, axes = plot.subplots(num_showed_imgs_y, num_showed_imgs_x, figsize=figsize)
    _ = fig.suptitle("Dataset images")

    _ = plot.setp(plot.gcf().get_axes(), xticks=[], yticks=[])
    for i, ax in enumerate(axes.flat):
        if i < len(dataset_picks):
            img = dataset_picks[i][0].int().permute(1, 2, 0).numpy()
            label_idx = dataset_picks[i][1]

            label_name = Marker(label_idx).name.capitalize().replace("_", " ")

            ax.imshow(img)
            ax.set_xlabel(label_name, fontsize=8)

    fig.tight_layout()
    plot.show()


def main():
    dataset = Dataset("./dataset/")

    random_25_idx = sample(range(0, len(dataset)), 25)

    dataset_picks = [dataset[idx] for idx in random_25_idx]

    show_images(dataset_picks)


if __name__ == "__main__":
    main()
