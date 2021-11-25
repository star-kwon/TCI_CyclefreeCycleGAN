# TCI_Cycle-free CycleGAN

## Paper
[Cycle-free CycleGAN using Invertible Generator for Unsupervised Low-Dose CT Denoising][paper link] (IEEE TCI, T. Kwon et al.)

[paper link]: https://ieeexplore.ieee.org/document/9622180


## Sample Data
Public dataset that we used was from [Low Dose CT Grand Challenge][aapm link].

[aapm link]: https://www.aapm.org/grandchallenge/lowdosect/


## Train 
You can use train.py for cycle-free CycleGAN.

For train.py,
training input & target data, test input & target data folder directory is required.


## Test
You can use inference.py for pre-trained cycle-free CycleGAN.

For inference.py,
test input & target data folder directory and pre-trained weight file directory is required.
