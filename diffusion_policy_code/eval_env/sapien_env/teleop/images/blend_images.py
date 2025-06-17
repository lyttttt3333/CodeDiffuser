from PIL import Image
from pathlib import Path
import numpy as np


def blend_images():
    current_path = Path(__file__).parent.resolve()
    images_path = list(Path(current_path).glob("*.png"))

    result = np.zeros_like(np.array(Image.open(images_path[0])), dtype=np.float32)
    images_num = len(images_path)
    for i in range(images_num):
        image = np.array(Image.open(images_path[i]))
        result += image / images_num
    result = Image.fromarray(result.astype(np.uint8))
    result.save(current_path / "blended_image.png")

    # base = np.array(Image.open(images_path[0]))
    # result = base.copy().astype(np.float32)
    # images_num = len(images_path)
    # visibility = 0.01
    # for i in range(1, images_num):
    #     image = np.array(Image.open(images_path[i]))
    #     result[..., :3] += (image[..., :3] - base[..., :3]) * visibility
    # result[..., 3] = 255
    # result = Image.fromarray(result.astype(np.uint8))
    # result.save(current_path / "blended_image.png")


if __name__ == "__main__":
    blend_images()
