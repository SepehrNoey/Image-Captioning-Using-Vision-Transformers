# Vision Transformer Image Captioning

This project implements image captioning using a Vision Transformer (ViT) architecture. The model consists of a Vision Encoder and a Text Decoder built from scratch by basic PyTorch blocks. The goal is to generate descriptive captions for images using a transformer-based approach.

## Demo

Here is a sample result showing how the model captions images:

|Image|Real Caption|Generated Caption|
|-----|-----|-----|
|![COCO_val2014_000000028194](https://github.com/user-attachments/assets/e2498162-9b16-48fe-bd2d-ce728fe76806)| A smiling woman in a wetsuit catches a wave on a surfboard|a man on a surfboard riding a wave in the ocean.|
|![COCO_val2014_000000396274](https://github.com/user-attachments/assets/ebe0e891-4a1a-4aaa-b27d-14dff22ede75)| A plant in a garden near a white building.|a vase of flowers sitting next to a window.|
|![COCO_val2014_000000140307](https://github.com/user-attachments/assets/1b81316b-3676-40c5-a4c9-9bdff6a9331e)| A group of people sitting around a table eating underneath an umbrella.|a group of people standing around a table with food.|
|![COCO_val2014_000000036484](https://github.com/user-attachments/assets/36ed0ed9-3945-4055-9aa0-49c7dcd5c1f2)| a cat sitting on a desk on a piece of lined paper next to a computer, pen and computer mouse.|a cat is laying on a laptop computer.|

However, the model generates many **inaccurate captions** too:
|Image|Real Caption|Generated Caption|
|-----|-----|-----|
|![COCO_val2014_000000084270](https://github.com/user-attachments/assets/8ae9a532-7f61-465a-87f6-ab7b9656dd71)| A busy airport with many people walking around.|a street with people walking in the rain.|
|![COCO_val2014_000000049763](https://github.com/user-attachments/assets/33306507-bcff-4a89-abf3-39dd7b2257dc)| The family running on the beach with many birds|a couple of people are sitting in the water|

---

## Model Architecture

The Vision Transformer (ViT) architecture is implemented using basic PyTorch blocks, including self-attention layers, multi-layer perceptrons (MLPs), and sinusoidal positional embeddings. The model consists of two main components:

- **Vision Encoder**: Extracts patches from input images and processes them through transformer layers.
- **Text Decoder**: Generates captions by attending to the encoded image features.

### Transformer Blocks

The core of the architecture is built using transformer blocks. Each block consists of:

- Self-attention mechanisms (Multi-Head)
- Layer normalization
- Multi-layer perceptron (MLP)
- Optional causal masking for the decoder

### Positional Embeddings

Positional embeddings are added to both image patches and input tokens to retain spatial information throughout the encoding and decoding process.

---

## Training

- The model was trained for **120 epochs** on a **Kaggle Notebook** using a **P100 GPU**.
- The dataset used was **MS-COCO 2014** training and validation partitions.
- The optimizer used was **Adam Optimizer** with a learning rate of `1e-4`, batch size of `128`, and images resized to `128x128`. Also used a `GradScaler` for stability in training.
- Below are the plots of the training loss during the epochs:

**Raw Training Loss**:

![raw_trainig_loss](https://github.com/user-attachments/assets/7747d6bc-f6e2-4f40-b10d-052e13fb3613)

**Averaged Training Loss** (Window size of 512 using `np.convolve`):

![averaged_training_loss](https://github.com/user-attachments/assets/6cb0396d-d412-491e-adb1-788057babab4)

---

## Credits
The code structure and training process was inspired and derived by **Luke Ditria** youtube page available at [here](https://www.youtube.com/@LukeDitria).