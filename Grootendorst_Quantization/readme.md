# Notes from Maarten Grootendorst's blog titled 'A Visual Guide to Quantization'

<https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization>

> Since this blog has so many INCREDIBLE annotations, I think it will be nice to take screenshots and annotate them. Full credits to the author!

Quantization: Reduce the precision of a model's parameter from **higher bit-widths** (e.g. 32-bit floating point) **to lower bit-widths** (e.g. 8-bit integers)

# INTRO

<img src="readme-images/bits1.png" alt="drawing" width="700"/>

<img src="readme-images/precision-range.png" alt="drawing" width="700"/>

Since there are 8 bits in a byte of memory, we can create a basic formula for most forms of floating point representation.

$$ \text{memory} = \dfrac{\text{nbr bits}}{8} \cdot \text{nbr params} $$

| bit-widths | Size needed |
| --- | --- |
| 64-bits | $\dfrac{64}{8} \cdot 70 \text{B} \approx 560 \text{GB}$ |
| 32-bits | $\dfrac{32}{8} \cdot 70 \text{B} \approx 280 \text{GB}$ |
| 16-bits | $\dfrac{16}{8} \cdot 70 \text{B} \approx 140 \text{GB}$ |

# Common DataTypes

| bit-widths | min | max | bits-sign | bits-exponent | bits-mantissa | bits-total | notes |
| ---        | --- | --- | ---       | ---           | ---           | ---        | ---   |
| FP32      | $-3.4e^{34}$ | $3.4e^{34}$ | 1 | 8 | 23 | 32 | our reference |
| FP16      | $-65504$ | $65504$ | 1 | 5 | 10 | 16 | range is quite smaller than FP32 |
| BP16 <br>  *(brain-float 16)*     | $-3.4e^{34}$ | $3.4e^{34}$ | 1 | 8 | 7 | 16 | *aka truncated FP32*. Uses the same amount of bits as FP16 but can take a wider range of values. Often used in deep learning applications |
| INT8 | $-127$ | $127$ | 1 | 0 | 7 | 8 | |

> In practice, we do not need to map the entire FP32 range  $[-3.4e^{34}, 3.4e^{34}]$ into INT8. We merely need to find a way to map the range of our data (the model's parameters) into INT8.

# Symmetric Quantization

Quantized value for zero in the floating-point space is exactly zero in the quantized space

**absmax quantization**

<img src="readme-images/symm1.png" alt="drawing" width="700"/>

# Asymmetric Quantization

Is not symmetric around zero. Instead, it maps the minimum (β) and maximum (α) values from the float range to the minimum and maximum values of the quantized range

**zero-point quantization**

<img src="readme-images/asymm1.png" alt="drawing" width="700"/>

# Range Mapping and Clipping

**Major downside of mapping to lower-bit representation : OUTLIERS**

<img src="readme-images/outliers1.png" alt="drawing" width="700"/>

Instead, *clip* certain values - we set different dynamic range - *all outliers get the same value*

e.g. **MANUALLY set dynamic range to [-5, 5]** all values outside that will either be mapped to -127 or to 127 regardless of their value:

- GOOD: quantization error of the non-outliers $\downarrow$
- BAD: quantization error of outliers $\uparrow$

# Calibration

Instead of manually selecting an arbitrary range like [-5, 5], selecting it such that

1. no. of values included $\uparrow$
1. quantization error $\downarrow$

> Since $ \text{no. of weights (billions)}  >> \text{no. of biases (millions)}$, main quantization effort is for weights, biases often kept at higher like INT16

<img src="readme-images/calib1.png" alt="drawing" width="700"/>

## Activations

Unlike weights and biases, activations vary with each input during inference. Two broad methods

1. Post-Training Quantization (PTQ): Quantization **after** training
1. Quantization Aware Training (QAT): Quantization **during** training/fine-tuning

# Post Training Quantization

- *Weights*: symm or asymm quantization
- *Activations*: *Dynamic* or *Static*

## Dynamic Quantization (during inference)

Data passes through model - collect *for each layer*, distributions of activations - collect zeropoint and scale factor - repeated each time data passes through a new layer

> Each layer has its own separate z and s values and therefore different quantization schemes!

## Static Quantization (beforehand)

**Calibration dataset** through model - collect distributions for all activations - calculate zeropoint and scale factor globally

> Dynamic is slower (because during inference) but accurate. Static is faster but less accurate

## 4-bit Quantization


### GPTQ (full model on GPU)

### GGUF (potentially offload layers on the CPU)
