# Color-Loss
PyTorch implementation of Color Loss in ["DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks"](https://arxiv.org/pdf/1704.02470.pdf)

## Package Requirements
* Python
* scipy
* numpy
* PyTorch

## Use Example
```python
from color_loss import Blur, ColorLoss


cl = ColorLoss()

# rgb example
blur_rgb = Blur(3)
img_rgb1 = torch.randn(4, 3, 40, 40)
img_rgb2 = torch.randn(4, 3, 40, 40)
blur_rgb1 = blur_rgb(img_rgb1)
blur_rgb2 = blur_rgb(img_rgb2)
print(cl(blur_rgb1, blur_rgb2))

# gray example
blur_gray = Blur(1)
img_gray1 = torch.randn(4, 1, 40, 40)
img_gray2 = torch.randn(4, 1, 40, 40)
blur_gray1 = blur_gray(img_gray1)
blur_gray2 = blur_gray(img_gray2)
print(cl(blur_gray1, blur_gray2))
```