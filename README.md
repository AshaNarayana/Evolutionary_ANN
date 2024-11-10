# CI - Training Neural Networks with Evolutionary Algorithms

## Introduction
Project Overview
This project explores an alternate approach to training artificial neural networks (ANNs) without using traditional backpropagation. The goal is to implement a method for optimizing ANNs using Genetic Algorithms, Evolution Strategies, and derivative-based techniques. By comparing these methods, we can understand their relative advantages, such as handling non-continuous and noisy error functions, and analyze their performance in terms of execution time, error rate, and model size. This approach also emphasizes examining the impact of architectural features like neuron count and weight limits on the overall performance and generalization of the ANN.

## Team

- Marina Bermúdez Granados (marina.bermudez.granados@estudiantat.upc.edu) <br />
- Asha Hosakote Narayana Swamy (asha.hosakote@estudiantat.upc.edu)


## Exercise Summary

Creating a full method for training an ANN as an alternate method to backprop-based techniques.

<b>Requirements:</b>
1. MLP or RBF network, one hidden layer
2. Synthetic data
3. One problem
4. One kind of ANN
5. One validation set

<b>Goal:</b> Compare GA/Evo Strategies against derivative methods (or all three)
## Getting Started

### Running script for the first time
These sections show how to create virtual environment for
our script and how to install dependencies
1. Open folder in terminal
  ```bash
  cd Evolutionary_ANN/
  ```

2. Create virtual env
  ```bash
  python3 -m venv venv/
  ```

3. Open virtual env
  ```bash
  source venv/bin/activate
  ```

4. Install required dependencies
  ```bash
  pip install -r requirements.txt
  ```
you can check if dependencies were installed by running next
command,it should print list with installed dependencies
  ```bash
  pip list
  ```

5. Close virtual env
  ```bash
  deactivate
  ```
All tests can be found at the directory `src/supervised_ml/experiments`. We indicate how to run each experiment (we assume that the virtual environment is set to run any python code from here.)

#### 1 ANN derivative without Keras
Implementation is found at directory `models/ANN_derivative.py`.
Experiments for execution and analysis phases are separated in files `experiments/experiment_ANN_derivative.py` and `experiments/Ann_derivative_analyse.py` respectively.

To run the experiment file to generate test file run the following code:
   ```bash
   cd ./experiments/experiment_ANN_derivative.py
   python3 experiment_ANN_derivative.py
   ```

We will see the results file in `results/ANN_derivative/Ann_derivative_test.csv`.

Then, to find the analysis run the following code
   ```bash
   python3 experiment_Ann_derivative_analyse.py
   ```


This will generate `results/ANN_derivative/best_ann_derivative_analyse.csv` and `results/ANN_derivative/worst_ann_derivative_analyse.csv`.
Where we can see best and worst parameters for the ANN.

#### 2 ANN derivative with Keras

Implementation is found at directory `models/ANN_derivative_keras.py`.
Experiments for execution and analysis phases are separated in files  `experiments/experiment_ANN_derivative_keras.py` and `experiments/experiment_Ann_derivative_keras_analyse.py` respectively.

To run the test file run the following code:
   ```bash
   cd ./experiments/experiment_ANN_derivative_keras.py
   python3 experiment_ANN_derivative_keras.py
   ```

We will see the results file in `results/ANN_derivative_keras/Ann_derivative_keras_test.csv`.

Then, to run the analysis run the following code
   ```bash
   python3 experiment_Ann_derivative_keras_analyse.py
   ```


This will generate `results/ANN_derivative_keras/best_ann_derivative_keras_analyse.csv` and `results/ANN_derivative_keras/worst_ann_derivative_keras_analyse.csv`.
Where we can see best and worst parameters for the ANN using Keras.

#### 3 ANN Evolutionary Algorithm with GA


## Resources 

### About Evolutionary Algorithms + ANN
https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e <br />
https://medium.com/@Data_Aficionado_1083/genetic-algorithms-optimizing-success-through-evolutionary-computing-f4e7d452084f <br />
https://www.comet.com/site/blog/train-neural-networks-using-a-genetic-algorithm-in-python-with-pygad/ <br />
https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad/ <br >
https://www.kaggle.com/code/zzettrkalpakbal/genetic-algorithm-tutorial-of-pygad

### About ANN
https://www.kaggle.com/code/androbomb/simple-nn-with-python-multi-layer-perceptron <br />
https://medium.com/@reddyyashu20/ann-python-code-in-keras-and-pytorch-d98841639ba0 <br />
https://www.analyticsvidhya.com/blog/2021/10/implementing-artificial-neural-networkclassification-in-python-from-scratch/
