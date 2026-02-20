from random import sample

import cv2.typing as cv2t
from cv2 import COLOR_BGR2RGB, cvtColor
from matplotlib import pyplot as plot

from dataset import Dataset, Marker


def show_images(dataset_picks: list[tuple[cv2t.MatLike, Marker]]):
    num_showed_imgs_x = 5
    num_showed_imgs_y = 5

    figsize = (10, 10)
    fig, axes = plot.subplots(num_showed_imgs_y, num_showed_imgs_x, figsize=figsize)
    _ = fig.suptitle("Dataset images")

    _ = plot.setp(plot.gcf().get_axes(), xticks=[], yticks=[])
    for i, ax in enumerate(axes.flat):
        if i < len(dataset_picks):
            img = cvtColor(dataset_picks[i][0], COLOR_BGR2RGB)
            label_idx = dataset_picks[i][1]

            label_name = label_idx.name.capitalize().replace("_", " ")

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
