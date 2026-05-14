# Story Sequencing Image Generation with GAN(Generative Adversarial Network)

This project implements a deep learning model for generating the next image in a sequence based on a story's visual and textual context. It uses a multimodal approach combining image sequences, text descriptions, and a Generative Adversarial Network (GAN) to predict future frames in visual stories.

## Overview

The model processes sequences of K images along with their corresponding text descriptions from stories, and learns to generate the next image in the sequence. The architecture includes:

- **Visual Encoder**: ResNet-50 based encoder for image features
- **Text Encoder**: LSTM-based encoder for text sequences
- **Fusion Module**: Combines visual and textual features per timestep
- **Sequence Model**: GRU for modeling temporal dependencies
- **Generator**: Transpose convolutional network for image generation
- **Discriminator**: PatchGAN discriminator for adversarial training

## Dataset

The project uses the [StoryReasoning dataset](https://huggingface.co/datasets/daniel3303/StoryReasoning) from Hugging Face, which contains visual stories with sequences of images and accompanying text.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Torchvision
- CLIP (for evaluation)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Key hyperparameters can be modified in the notebook:

- `K`: Sequence length (number of input frames)
- `IMG_SIZE`: Image resolution (64x64)
- `BATCH_SIZE`: Training batch size
- `EPOCHS`: Number of training epochs
- `LR`: Learning rate

## Usage

### Training

1. Open `DNN.ipynb` in Jupyter/VS Code
2. Run all cells to train the model
3. Checkpoints will be saved in `checkpoints_seq/` directory
4. Generated samples during training are saved in `results/samples_seq/`
5. Training plots are saved in `results/plots/`

### Evaluation

The model uses CLIP similarity to evaluate how well generated images match the story context.

## Project Structure

```
.
├── DNN.ipynb              # Main training notebook
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── dataset/               # Dataset utilities
├── checkpoints_seq/       # Saved model checkpoints
├── results/
│   ├── samples_seq/       # Generated image samples
│   └── plots/             # Training loss plots
└── README.md              # This file
```

## Model Architecture

### Input Processing
- Images are resized to 64x64 and normalized
- Text is tokenized using BERT tokenizer and padded/truncated to 32 tokens
- Sequences of K (image, text) pairs are processed

### Training
- Adversarial loss between generator and discriminator
- L1 reconstruction loss for pixel-level accuracy
- CLIP-based evaluation for semantic alignment

## Results

After training, the model can generate coherent next images in story sequences. Check the `results/` directory for:

- Sample generations during training
- Loss curves for generator and discriminator
- CLIP similarity scores over time

## Troubleshooting

### SSL Certificate Issues
If you encounter SSL verification errors when downloading models or datasets, the notebook includes SSL context fixes for macOS.

### Memory Issues
- Reduce `BATCH_SIZE` if running out of GPU/CPU memory
- Decrease `SUBSET_SIZE` for quicker experimentation

### Dataset Loading
Ensure you have sufficient disk space for downloading the StoryReasoning dataset (~2GB).

## License

This project is for educational and research purposes. Please check the licenses of the datasets and models used.

## Acknowledgments

- StoryReasoning dataset by Daniel3303
- Based on multimodal sequence modeling research
- Uses pretrained models from Hugging Face and PyTorch

## SUMMARY

This project implements a multimodal GAN for generating the next image in a sequence of story frames, combining visual and textual inputs. The model uses a sequence of K images and their associated texts to predict future images, evaluated with CLIP similarity for semantic alignment. Key components include ResNet-based visual encoding, LSTM text encoding, GRU sequence modeling, and a GAN with adversarial and L1 losses. Training involves processing the StoryReasoning dataset with configurable hyperparameters for quick experimentation or full training. Results include generated samples, loss plots, and checkpoints saved for inference. The implementation handles SSL issues for macOS and supports CPU/GPU training, making it accessible for research and development in multimodal story understanding.
