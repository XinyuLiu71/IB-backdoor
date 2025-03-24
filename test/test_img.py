from PIL import Image
import numpy as np
img = "ddpm-cifar10-32/samples/sample_epoch_44.png"
img = Image.open(img)
print(f"img.max(): {np.array(img).max()}, img.min(): {np.array(img).min()}")
