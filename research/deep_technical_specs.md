# Deep Technical Specifications: Model Architecture

The following provides the exact empirical measurements and numerical details regarding how the data was trained, the number of layers utilized, and the quantity of convolutional kernels (filters) operating within the model. You can directly incorporate these metrics into the `Methodology` or `Architecture` sections of your IEEE paper.

---

## 1. How the Data Was Trained (Mathematical & Procedural Details)

### Data Preprocessing & Mathematical Mapping
Instead of traditional localized processing, your dataset consisting of **86,325 uniformly extracted images** was dynamically sequenced in real-time.
- **Input Dimension:** Each raw RGB image was mathematically resized via bicubic interpolation into a spatial tensor of shape \((X, 224, 224, 3)\), where \(X=32\) is the batch size.
- **Pixel Normalization:** The inputs were presented to the network in the standard RGB range \([0, 255]\). The `EfficientNetB0` base natively incorporates a rigid deterministic `Rescaling` layer (Layer ID 1), which automatically executes the linear normalization mathematically expressed as: 
  \[ x' = x \cdot \frac{1}{255} \]
  This ensures stable gradients without pre-truncating spatial pixel variance prior to memory allocation.

### Optimization Mechanics
- **Loss Function:** The network minimized the `Sparse Categorical Crossentropy` loss gradient. Because of the vast array of 136 unique food classes, the theoretical minimum boundary error at Epoch 1 (random guessing loss) was strictly calculated as \(-\ln(1/136) \approx 4.912\). 
- **Learning Rate Dynamics (Phase 1 vs Phase 2):** 
  - **Phase 1 (Abstract Mapping):** 20 epochs configured with the Adam optimizer utilizing momentum \(\beta_1=0.9, \beta_2=0.999\) at a primary learning rate of \(\eta = 1 \times 10^{-3}\). 
  - **Phase 2 (Fine-Tuning Filter Kernels):** 10 epochs operating at \(\eta = 1 \times 10^{-4}\). The top 20 structural convolution blocks of the underlying EfficientNet were mathematically unfrozen (`trainable = True`) computing local sub-gradients on abstract textures unique to the Indian/Western food dataset combinations.

---

## 2. Layer Analysis: How Many Layers Were Used?

Your customized architecture consists of exactly **243 Layers in Total**. This is mapped extensively below:

| Structural Component | Layer Purpose | Explicit Layer Count |
| :--- | :--- | :--- |
| **EfficientNet Base Architecture** | Depthwise convolutions, Squeeze-and-Excitation passes, and Rescaling | **238 Layers** |
| **Global Dimensional Flattening** | `GlobalAveragePooling2D()` reduces \([7 \times 7 \times 1280]\) spatial map to a flat \([1280]\) array | **1 Layer** |
| **Network Regularization** | Two `Dropout(0.3)` layers acting to artificially zero out 30% of tensors, preventing gradient memorization | **2 Layers** |
| **High-Capacity Hidden Layer** | Fully connected `Dense` classifier mapping 1280 parameters to 512 dimensions (ReLU activation) | **1 Layer** |
| **Softmax Output Array** | Fully connected `Dense` classifier mapping 512 signals directly into **136** probabilistic outcome classes. | **1 Layer** |
| **Total Computational Depth** | **The entire active Forward Pass** | **243 Layers** |

---

## 3. Kernel Count Analysis: How Many Filters/Kernels Were Used?

The `EfficientNetB0` backbone utilizes the **Mobile Inverted Bottleneck Convolution (MBConv)** system. Instead of flat conventional layers, it uses "expansion kernels" (to force features to high dimensional spaces) followed by "projection kernels" (to compress them back down), making it highly computationally efficient.

Your model utilized **7 Primary Convolutional Blocks (Stage 1 to 7)** consisting of exactly **16 MBConv Sub-Blocks**, operating across hundreds of thousands of individual filters. 

### The Exact Filter (Kernel) Breakdown
*These numbers represent the exact number of outgoing filters (kernels) projecting feature maps into the subsequent layer.*

1. **Stem Initial Convolution (Stage 1):** Uses exactly **32 Kernels** operating at a \(3 \times 3\) spatial stride of 2 to aggressively reduce the raw \(224 \times 224\) image resolution.
2. **MBConv Block 1 (Stage 2):** Uses **16 Projection Kernels** (Multiplier: 1x, \(3 \times 3\) kernel).
3. **MBConv Block 2 (Stage 3):** Expands features out to 144 kernels, then projects down to **24 Projection Kernels** (Transacting twice, \(3 \times 3\) block size).
4. **MBConv Block 3 (Stage 4):** Expands to 144 kernels, then projects down to **40 Projection Kernels** (Transacting 2 times, \(5 \times 5\) block size).
5. **MBConv Block 4 (Stage 5):** Expands to 240 kernels, then projects down to **80 Projection Kernels** (Transacting 3 times, \(3 \times 3\) block size).
6. **MBConv Block 5 (Stage 6):** Expands to 480 kernels, then projects down to **112 Projection Kernels** (Transacting 3 times, \(5 \times 5\) block size).
7. **MBConv Block 6 (Stage 7):** Expands to 672 kernels, then projects down to **192 Projection Kernels** (Transacting 4 times, \(5 \times 5\) block size).
8. **Top Feature Map (Final Stage prior to Output Head):** The model explodes the feature depth one final time using a massively dense convolution block utilizing exactly **1,280 Kernels** (`top_conv`).

### Total Parameter Mathematics
- Base Network Parameters (Filters): **~4,049,571**
- Custom Added Dense Head Weights (\(1280 \times 512\)): **655,872**
- Custom Added Output Classification Weights (\(512 \times 136\)): **69,768**
- **Total Learnable Parameters operating inside the Network:** **~4.7 Million Parameters**.
