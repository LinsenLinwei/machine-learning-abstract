# Quantized SGD in federated learning: communication, optimization and generalization (PyTorch)

Implementation of the paper : [Quantized SGD in federated learning: communication, optimization and
generalization](https://www.).

##Abstract
```
In this paper, we are interested in the interplay among some key factors of the widely-used distributed SGD with 
quantization, including quantization level, optimization error and generalization performance. For convex objectives, 
our main results show that there exist several trade-offs among communication efficiency, optimization error
 and generalization performance.
```
Experiments are produced on MNIST, Fashion MNIST and CIFAR-10 (both IID). 

Since the purpose of these experiments are to illustrate the effectiveness of 
the distributed learning paradigm, only simple models such as convex model (Logistic Regression) 
and non-convex (CNN and Resnet-18) are used.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar10.

## Running the experiments

* To run the experiment with MNIST on Logistic Regression:
```
python main.py --model=LogisticRegression --dataset=mnist --epochs=50 --quantizer=sgd --lr=0.1 --momentum=0.5
python main.py --model=LogisticRegression --dataset=mnist --epochs=50 --quantizer=br --p=0.8 --lr=0.1 --momentum=0.5
python main.py --model=LogisticRegression --dataset=mnist --epochs=50 --quantizer=br --p=0.5 --lr=0.1 --momentum=0.5
python main.py --model=LogisticRegression --dataset=mnist --epochs=50 --quantizer=br --p=0.2 --lr=0.1 --momentum=0.5
```
* To run the experiment with fashionmnist on Logistic Regression:
```
python main.py --model=LogisticRegression --dataset=fashionmnist --epochs=50 --quantizer=sgd --lr=0.01 --momentum=0.8
python main.py --model=LogisticRegression --dataset=fashionmnist --epochs=50 --quantizer=br --p=0.8 --lr=0.01 --momentum=0.8
python main.py --model=LogisticRegression --dataset=fashionmnist --epochs=50 --quantizer=br --p=0.5 --lr=0.01 --momentum=0.8
python main.py --model=LogisticRegression --dataset=fashionmnist --epochs=50 --quantizer=br --p=0.2 --lr=0.01 --momentum=0.8
```
* To run the experiment with fashionmnist on CNN:
```
python main.py --model=net --dataset=fashionmnist --epochs=50 --quantizer=sgd --lr=0.01 --momentum=0.8
python main.py --model=net --dataset=fashionmnist --epochs=50 --quantizer=br --p=0.8 --lr=0.01 --momentum=0.8
python main.py --model=net --dataset=fashionmnist --epochs=50 --quantizer=br --p=0.5 --lr=0.01 --momentum=0.8
python main.py --model=net --dataset=fashionmnist --epochs=50 --quantizer=br --p=0.2 --lr=0.01 --momentum=0.8
```
* To run the experiment with CIFAR-10 on CNN:
```
python main.py --model=resnet18 --dataset=cifar10 --epochs=50 --quantizer=sgd --lr=0.1 --momentum=0.8
python main.py --model=resnet18 --dataset=cifar10 --epochs=50 --quantizer=br --p=0.8 --lr=0.1 --momentum=0.8
python main.py --model=resnet18 --dataset=cifar10 --epochs=50 --quantizer=br --p=0.5 --lr=0.1 --momentum=0.8
python main.py --model=resnet18 --dataset=cifar10 --epochs=50 --quantizer=br --p=0.2 --lr=0.1 --momentum=0.8
```
You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fashionmnist', 'cifar10'
* ```--model:```    Default: 'LogisticRegression'. Options: 'LogisticRegression', 'net', 'resnet18'
* ```--gpu:```      Default: runs on GPU.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--seed:```     Random Seed. Default set to 1.

#### Parameters

* ```--num_users:```Number of users. Default is 4.
* ```--epochs:``` Number of local training epochs in each user. Default is 50.
* ```--batch-size:``` Batch size of local updates in each user. Default is 128.



## Further Readings
### Papers:
* [Train faster, generalize better: Stability of stochastic gradient descent](http://proceedings.mlr.press/v48/hardt16.pdf)
* [QSGD: Communication-efficient SGD via gradient quantization and encoding](https://papers.nips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf)

### Blog Posts:
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)

## Citation
Please cite our paper if you found useful.
```latex
@article{
}
```
