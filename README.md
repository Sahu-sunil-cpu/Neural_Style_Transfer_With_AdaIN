
# Neural Style Transfer with AdaIN

<p align="center">
  <b>Neural Style Transfer using Adaptive Instance Normalization (AdaIN)</b>
</p>

<p align="center">
  PyTorch-based implementation for transferring the artistic style of one image onto the content of another.
</p>

---

## Overview

This project implements **Neural Style Transfer using Adaptive Instance Normalization (AdaIN)**.

The system takes two images:

- A **content image** that provides the structure and semantic information.
- A **style image** that provides the artistic appearance.

A pretrained VGG encoder extracts feature representations from both images. AdaIN then aligns the channel-wise statistics of the content features with those of the style features. A trained decoder reconstructs the resulting feature representation into a stylized image.

The project also includes a Flask web application for performing style transfer through a browser.

---

## Features

- Adaptive Instance Normalization (AdaIN)
- Pretrained VGG encoder
- Custom-trained decoder
- Arbitrary content-style combinations
- Adjustable style strength using `alpha`
- PyTorch-based training pipeline
- CUDA GPU acceleration
- CPU fallback for inference
- Flask web interface
- Model checkpointing
- Configurable training parameters

---

## Architecture

```text
                    Content Image
                          │
                          ▼
                    VGG Encoder
                          │
                          ▼
                  Content Features
                          │
                          │
                          ▼
                        AdaIN
                          ▲
                          │
                          │
                   Style Features
                          ▲
                          │
                    VGG Encoder
                          ▲
                          │
                     Style Image
                          │
                          ▼
                Stylized Feature Map
                          │
                          ▼
                      Decoder
                          │
                          ▼
                  Stylized Image
````

---

## How AdaIN Works

Adaptive Instance Normalization transfers the statistical characteristics of the style feature representation to the content feature representation.

The core equation is:

[
AdaIN(x,y) =
\sigma(y)
\left(
\frac{x-\mu(x)}
{\sigma(x)}
\right)
+\mu(y)
]

where:

* `x` = content feature
* `y` = style feature
* `μ` = channel-wise mean
* `σ` = channel-wise standard deviation

The implementation performs:

```python
stylized_feats = adaptive_instance_normalization(
    content_feats,
    style_feats
)
```

The content features are first normalized and then rescaled using the mean and standard deviation of the style features.

---

## Style Strength

The project supports an `alpha` parameter that controls the strength of the transferred style.

The final feature representation is:

[
t =
\alpha \cdot AdaIN(c,s)
+
(1-\alpha)\cdot c
]

where:

* `α = 0` → original content
* `α = 1` → full style transfer
* `0 < α < 1` → partial style transfer

This allows the user to control how strongly the artistic style is applied.

---

## Training

The VGG encoder is kept frozen while the decoder is trained.

The training pipeline is:

```text
Content Image
      │
      ▼
 VGG Encoder
      │
      ▼
Content Features
      │
      │
      ▼
     AdaIN
      ▲
      │
      │
Style Features
      ▲
      │
 VGG Encoder
      ▲
      │
 Style Image
      │
      ▼
Target Features
      │
      ▼
   Decoder
      │
      ▼
Generated Image
      │
      ▼
 VGG Encoder
      │
      ▼
Generated Features
```

The decoder learns to reconstruct an image from the transformed feature representation.

---

## Loss Function

The total loss consists of content loss and style loss:

[
L = L_{content} + L_{style}
]

### Content Loss

The generated feature representation is compared with the AdaIN target:

```python
loss_c = mse_loss(
    g_feats[-1],
    t
) * content_weight
```

The project uses Mean Squared Error:

```python
torch.nn.MSELoss()
```

### Style Loss

The style loss compares the mean and standard deviation of generated and style feature maps:

```python
g_mean, g_std = calc_mean_std(g_f)
s_mean, s_std = calc_mean_std(s_f)

loss_s += (
    mse_loss(g_mean, s_mean)
    + mse_loss(g_std, s_std)
)
```

The final loss is:

[
L = L_c + L_s
]

---

## Dataset

The training setup uses separate content and style image datasets.

| Dataset | Number of Images |
| ------- | ---------------: |
| Content |            8,000 |
| Style   |           40,000 |

The training configuration uses:

```text
Batch size = 4
```

Therefore, the content dataset produces:

[
8000 / 4 = 2000
]

batches per epoch.

With 10 epochs:

[
2000 \times 10 = 20,000
]

training iterations.

Because the training loop uses:

```python
zip(content_dataloader, style_dataloader)
```

the number of iterations per epoch is determined by the smaller dataloader.

---

## Training Configuration

| Parameter           |    Value |
| ------------------- | -------: |
| Batch Size          |        4 |
| Epochs              |       10 |
| Learning Rate       |   `1e-4` |
| Learning Rate Decay |   `5e-5` |
| Content Weight      |    `1.0` |
| Style Weight        |      `5` |
| Content Size        |    `512` |
| Style Size          |    `512` |
| Final Size          |    `256` |
| Save Interval       | 2 epochs |

---

## Model Checkpoints

The training script periodically saves the decoder and optimizer states.

With:

```python
save_interval = 2
```

and:

```python
epochs = 10
```

the decoder checkpoints are:

```text
decoder_2.pth
decoder_4.pth
decoder_6.pth
decoder_8.pth
decoder_10.pth
```

The corresponding optimizer checkpoints are:

```text
optimizer_2.pth
optimizer_4.pth
optimizer_6.pth
optimizer_8.pth
optimizer_10.pth
```

The `.pth` files contain PyTorch model parameters or optimizer state dictionaries.

---

## Inference

After training, the decoder checkpoint can be loaded for inference:

```python
decoder.load_state_dict(
    torch.load(
        "experiment/trial1/decoder_6.pth",
        map_location=device
    )
)
```

The inference pipeline is:

```text
Content Image
      │
      ▼
VGG Encoder
      │
      ▼
Content Features
      │
      ├─────────────────┐
      │                 │
      ▼                 ▼
     AdaIN ◄────── Style Features
      │
      ▼
Alpha Blending
      │
      ▼
Trained Decoder
      │
      ▼
Stylized Image
```

Once the decoder has been trained, it can be reused with different content and style images without retraining for every image pair.

---

## Web Application

The project includes a Flask-based web interface.

The application supports:

* Content image upload
* Style image upload
* Style strength control
* Image preview
* Stylized image generation
* Example images
* Error handling

The application automatically selects the available device:

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
```

If CUDA is available, inference runs on the NVIDIA GPU. Otherwise, the application falls back to CPU.

---

## Project Structure

```text
Neural_Style_Transfer_With_adaIN/
│
├── train.py
├── app.py
├── vgg_normalised.pth
├── requirements.txt
│
├── nst/
│   └── utils/
│       ├── models.py
│       └── utils.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
│
├── examples/
│   ├── brad_pitt.jpg
│   ├── picasso_seated_nude_hr.jpg
│   └── ...
│
└── experiment/
    └── trial1/
        ├── args.txt
        ├── decoder_2.pth
        ├── decoder_4.pth
        ├── decoder_6.pth
        ├── decoder_8.pth
        ├── decoder_10.pth
        └── output_*.png
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sahu-sunil-cpu/Neural_Style_Transfer_With_adaIN.git

cd Neural_Style_Transfer_With_adaIN
```

### 2. Create a Conda Environment

```bash
conda create -n adain python=3.12
```

Activate it:

```bash
conda activate adain
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running Training

Example:

```bash
python train.py \
    --batch_size 4 \
    --epochs 10 \
    --experiment trial1
```

The trained checkpoints will be stored under:

```text
experiment/trial1/
```

---

## Running the Web Application

Start the Flask application:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

Upload a content image and a style image, adjust the style strength, and run the style-transfer process.

---

## GPU Acceleration

The project supports NVIDIA CUDA through PyTorch.

Check CUDA availability:

```python
import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Example:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 2050
```

GPU utilization can be monitored using:

```bash
nvidia-smi
```

On Windows PowerShell:

```powershell
while ($true) {
    nvidia-smi
    Start-Sleep -Seconds 1
}
```

---

## Memory Optimization

Neural Style Transfer with VGG and a decoder can require significant memory.

For lower-memory environments, inference can use a smaller resolution:

```python
transforms.Resize((256, 256))
```

instead of:

```python
transforms.Resize((512, 512))
```

This reduces memory consumption but also reduces the spatial resolution of the generated output.

Inference can also use:

```python
with torch.no_grad():
    ...
```

to avoid storing gradients.

---

## Technical Highlights

### Adaptive Instance Normalization

AdaIN enables arbitrary style transfer by matching the channel-wise statistics of content and style feature representations.

### Encoder-Decoder Architecture

The VGG encoder extracts feature representations, while the decoder reconstructs the stylized image.

### Frozen Encoder

The VGG encoder remains fixed during decoder training, allowing the decoder to learn the reconstruction process without updating the feature extractor.

### Arbitrary Style Transfer

After training, the same decoder can be used with different content and style images without retraining for each pair.

### Style Strength Control

The `alpha` parameter provides direct control over the intensity of the transferred style.

---

## Technologies Used

| Category          | Technology            |
| ----------------- | --------------------- |
| Language          | Python                |
| Deep Learning     | PyTorch               |
| Computer Vision   | Torchvision           |
| Image Processing  | Pillow                |
| Architecture      | AdaIN                 |
| Feature Extractor | VGG                   |
| Web Framework     | Flask                 |
| Frontend          | HTML, CSS, JavaScript |
| Optimizer         | Adam                  |
| Loss Function     | MSE                   |
| GPU Acceleration  | CUDA                  |

---

## Future Improvements

* High-resolution inference
* Faster CPU inference
* GPU cloud deployment
* Batch style transfer
* Multiple pretrained decoder checkpoints
* Additional style presets
* Downloadable generated images
* Model quantization
* Asynchronous inference
* Improved web optimization

---

## References

The implementation is based on:

**Huang, X. & Belongie, S. — Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization**

The core approach transfers style by aligning channel-wise feature statistics between content and style representations.

---

## Author

**Sunil Kumar Sahu**

B.Tech — Computer Science & Engineering

GitHub:
https://github.com/Sahu-sunil-cpu

---

## License

This project is intended for educational and portfolio purposes.

