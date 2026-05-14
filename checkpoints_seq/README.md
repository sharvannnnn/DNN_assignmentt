Checkpoints Directory Information

This directory contains saved model checkpoints from the training process of the Story Sequence Image Generation GAN.

File Naming Convention:
- epoch_{epoch_number}.pt: Checkpoint saved at the end of each training epoch

Contents of Each Checkpoint:
- G: State dictionary of the Generator model (StorySeqGAN)
- D: State dictionary of the Discriminator model
- g_opt: State dictionary of the Generator optimizer
- d_opt: State dictionary of the Discriminator optimizer

Usage:
- To resume training, load the checkpoint and restore the models and optimizers
- For inference, load only the Generator (G) state dictionary
- Checkpoints are saved in PyTorch format (.pt)

Training Configuration:
- Images resized to 64x64
- Sequence length K=3
- Batch size=4
- Learning rate=1e-4
- Adversarial loss weight=1.0, L1 loss weight=10.0

Note: Checkpoints are generated during training. If this directory is empty, training has not been completed yet.