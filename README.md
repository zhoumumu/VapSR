#  VapSR
Efficient Image Super-Resolution using Vast-Receptive-Field Attention(ECCVW 2022). [Paper link](https://arxiv.org/pdf/)

The attention mechanism plays a pivotal role in designing advanced super-resolution (SR) networks. In this work, we design an efficient SR network by improving the attention mechanism. We start from a simple pixel attention module and gradually modify it to achieve better super-resolution performance with reduced parameters. The specific approaches include: (1) increasing the receptive field of the attention branch, (2) replacing large dense convolution kernels with depthwise separable convolutions, and (3) introducing pixel normalization. These approaches paint a clear evolutionary roadmap for the design of attention mechanisms. Based on these observations, we propose VapSR, the Vast-receptive-field Pixel attention network. Experiments demonstrate the superior performance of VapSR. VapSR outperforms the present lightweight networks with even fewer parameters. And the light version of VapSR can use only 21.68% and 28.18% parameters of IMDB and RFDN to achieve similar performances to those networks.

<!-- ![](https://raw.githubusercontent.com/zhoumumu/VapSR/main/network.jpg) -->
![](./network.jpg)

Performance of X4 scale (PSNR / SSIM on Y channel):
| <sub> model </sub> | <sub> Pararms[K] </sub> | <sub> Set5 </sub> | <sub> Set14 </sub> |  <sub> B100 </sub> | <sub> Urban100 </sub> |
|  :----:  | :----:  |  :----:  | :----:  |  :----:  | :----:  |
|  VapSR-S  | 155 | 32.14/0.8951 | 28.64/0.7826 | 27.60/0.7373 | 26.05/0.7852 |
|  VapSR  | 342 | 32.38/0.8978 | 28.77/0.7852 | 27.68/0.7398 | 26.35/0.7941 |


## Code Directions
The code is constructed based on [BasicSR](https://github.com/XPixelGroup/BasicSR). You can prepare the datasets needed follow the [document](https://github.com/XPixelGroup/BasicSR-docs).

To keep your workspace clean and simple, only `test.py`, `train.py` and `your_arch.py` are needed then you are good to go.

This line in `test.py` and `train.py` enables them to register your arch:
```
from vapsr_beta import vapsr_beta
```

Clone this github repo to make the reproducing.
```
>>> git clone https://github.com/zhoumumu/VapSR.git
>>> cd VapSR
```


## Testing
The provided pth files contain two models each, one is the EMA model and the other is the in-training-progress model with larger parameters. Our testing results are produced by the EMA models.

To load them, you can either remove the other model in the file and rename the key of the EMA model from 'params_ema' to 'params'. You need to make the process for every model you test.

Or you can change the BasicSR’s code in the position of `BasicSR/basicsr/models/sr_model.py line29` from
```
param_key = self.opt['path'].get('param_key_g', 'params')
```
to
```
param_key = self.opt['path'].get('param_key_g', 'params_ema')
```
Note this will change the behavior of the consuming training (load and train). Make sure you know which model you need and which model you loaded.

You can run the testing demo with
```
>>> python code/test.py -opt options/test/VapSR_X4.yml
```


## Training
Reproduce with
```
>>> python code/train.py -opt options/train/VapSR_X4.yml
```