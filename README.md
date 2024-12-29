# Knowledge Distillation from VGG16 to Mobilenet

- Teacher model: VGG16  
- Student model: Mobilenet

## Requirements

- Python 3.10  
- PyTorch 2.2.1+cu121  
- ptflops

## Usage

### 1. Models

#### VGG16
VGG16, a renowned CNN architecture from the University of Oxford, excels in image classification with its 16 layers and simple structure. It comprises 13 convolutional and 3 fully connected layers, employing 3x3 filters and 2x2 max-pooling. Despite its effectiveness, its size and depth can be computationally intensive.

#### MobileNet
MobileNet, developed by Google, is tailored for mobile and embedded devices, featuring 28 layers and innovative depthwise separable convolutions. This reduces parameters and computational complexity while maintaining performance. MobileNet adapts to various input sizes and is widely used for transfer learning, albeit potentially sacrificing some accuracy for efficiency.

### 2. Dataset
We have used the CIFAR-100 dataset.  
- It contains 60,000 color images.  
- Images are of size 32x32 pixels.  
- The dataset is organized into 100 classes, each containing 600 images.  
- There are 50,000 training images and 10,000 testing images.  
- Each image is labeled with one of the 100 fine-grained classes.

### 3. Train the Model

To train the VGG16 model:
```bash
# use gpu to train vgg16
$ python train.py -net vgg16 -gpu
```

To train the MobileNet model:
```bash
# use gpu to train mobilenet
$ python train.py -net mobilenet -gpu
```

To perform knowledge distillation from the trained VGG16 to the MobileNet model:
```bash
# use gpu to train mobilenet
$ python knowledge_distillation_train.py -gpu -teacher path_to_best_vgg16_weights_file -student path_to_best_mobilenet_weights_file
```
The weights file with the best accuracy would be written to the disk with a name suffix 'best' (default in the checkpoint folder).

### 4. Test the Model

Test the VGG16 model:
```bash
$ python test.py -net vgg16 -weights path_to_best_vgg16_weights_file
```

Test the MobileNet model:
```bash
$ python test.py -net mobilenet -weights path_to_best_mobilenet_weights_file
```

Test the knowledge-distilled MobileNet model:
```bash
$ python knowledge_distillation_test.py -gpu -weights path_to_best_knowledge_distilled_mobilenet_weights_file
```

## Implementation Details and References

- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)  
- MobileNet: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)  
- Hyperparameter settings: [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2), which is init lr = 0.1 divided by 5 at 60th, 120th, 160th epochs, trained for 200 epochs with batchsize 128 and weight decay 5e-4, Nesterov momentum of 0.9.  
- Code reference: [GitHub: weiaicunzai](https://github.com/weiaicunzai/pytorch-cifar100)

## Results

### Previous Results

| Dataset  | Network           | Learning Rate | Batch Size | Size (MB) | Params  | Top-1 Err | Top-5 Err | Time (ms) per Inference Step (GPU) | Time (ms) per Inference Step (CPU) | FLOPs   |
|:--------:|:-----------------:|:-------------:|:----------:|:---------:|:-------:|:---------:|:---------:|:--------------------------------:|:---------------------------------:|:-------:|
| CIFAR-100| VGG16             | 0.1           | 128        | 136.52    | 34.0M   | 27.77     | 10.12     | 177.2584                         | 10.7589                           | 334.14  |
| CIFAR-100| MobileNet         | 0.1           | 128        | 24.03     | 3.32M   | 33.06     | 10.15     | 57.6361                          | 9.0793                            | 48.32   |
| CIFAR-100| Knowledge Distilled MobileNet | 0.1 | 128 | 24.03 | 3.32M | 32.61 | 10.26 | 56.7409 | 9.6162 | 48.32 |
| CIFAR-100| Knowledge Distilled MobileNet | 0.001 | 64 | 24.03 | 3.32M | 32.16 | 10.83 | 58.2087 | 9.0350 | 48.32 |

### Results from Varying Soft Target Weight for Every Alternating Epoch

| Experiment No. | Loss Function                                           | Soft Target Weight | Cross Entropy Loss Weight | Top-1 Error Rate | Top-5 Error Rate | Top-1 Accuracy | Top-5 Accuracy | Parameters | Time (ms) per Inference (GPU) | FLOPs   |
|:--------------:|:------------------------------------------------------:|:------------------:|:-------------------------:|:----------------:|:----------------:|:--------------:|:--------------:|:----------:|:-----------------------------:|:-------:|
| 1              | Epoch%2==0: distillation loss, else cross-entropy loss                                     | 0.00001           | 1                         | 34.94           | 11               | 65.06          | 89             | 3.32M      | 9.4931                         | 48.32   |
| 2              | Epoch%2==0: distillation loss, else cross-entropy loss | 0.0001            | 1                         | 34.76           | 10.83            | 65.24          | 89.17          | 3.32M      | 5.0192                         | 48.32   |
| 3              | Epoch%2==0: distillation loss, else cross-entropy loss                                     | 0.001             | 1                         | 32.61           | 10.12            | 67.39          | 89.59          | 3.32M      | 5.1017                         | 48.32   |
| 4              | Epoch%2==0: distillation loss, else cross-entropy loss                                    | 0.01              | 1                         | 34.87           | 10.71            | 65.13          | 89.29          | 3.32M      | 4.8188                         | 48.32   |
| 5              | Epoch%2==0: distillation loss, else cross-entropy loss                                    | 0.1               | 1                         | 34.64           | 10.69            | 65.36          | 89.31          | 3.32M      | 3.9408                         | 48.32   |
| 6              | Epoch%2==0: distillation loss, else cross-entropy loss                                     | 0.5               | 1                         | 36.04           | 11.5             | 63.96          | 88.5           | 3.32M      | 4.3824                         | 48.32   |
| 7              | Epoch%2==0: distillation loss, else cross-entropy loss                                    | 1                 | 1                         | 33.46           | 11.78            | 66.54          | 88.22          | 3.32M      | 10.7554                        | 48.32   |

#### Details and Inferences

The alternating epoch experiment aimed to balance the benefits of distillation loss (soft targets) with cross-entropy loss (hard targets). Key observations include:

1. **Performance Gains:**
   - Alternating between distillation loss and cross-entropy loss (Experiment 2) improves accuracy compared to exclusive cross-entropy loss (Experiment 1).
   - This suggests that the student model benefits from the additional knowledge provided by soft targets during specific epochs.

2. **Sensitivity to Soft Target Weights:**
   - Experiment 3 (soft target weight = 0.001) achieves the best Top-1 Accuracy (67.39%) and the lowest Top-1 Error Rate (32.61%). This highlights the importance of fine-tuning the soft target weight for optimal results.
   - Increasing the soft target weight beyond a certain point (e.g., Experiment 6 with weight = 0.5) can lead to performance degradation, likely due to over-reliance on soft targets.

3. **Efficiency Considerations:**
   - Lower soft target weights generally result in reduced inference times on the GPU, as seen in Experiments 3 and 4.

4. **Trade-offs Between Accuracy and Inference Time:**
   - While higher soft target weights can improve accuracy, they may increase inference time, as demonstrated in Experiments 1 and 7.

Overall, the results confirm that alternating epochs with a balanced loss function approach can enhance the student model's performance while maintaining computational efficiency.

