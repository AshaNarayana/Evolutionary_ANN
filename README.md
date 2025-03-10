# CI - Training Neural Networks with Evolutionary Algorithms

## Introduction
Project Overview
This project explores an alternate approach to training artificial neural networks (ANNs) without using traditional backpropagation. The goal is to implement a method for optimizing ANNs using Genetic Algorithms, Evolution Strategies, and derivative-based techniques. By comparing these methods, we can understand their relative advantages, such as handling non-continuous and noisy error functions, and analyze their performance in terms of execution time, error rate, and model size. This approach also emphasizes examining the impact of architectural features like neuron count and weight limits on the overall performance and generalization of the ANN.

## Team

- Marina Bermúdez Granados (marina.bermudez.granados@estudiantat.upc.edu) <br />
- Asha Hosakote Narayana Swamy (asha.hosakote@estudiantat.upc.edu)

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
You can check if dependencies were installed by running next
command,it should print list with installed dependencies
  ```bash
  pip list
  ```

5. Close virtual env
  ```bash
  deactivate
  ```

### Generate synthetic data
In data/synthetic_data_generator.py, we created a SyntheticDataGenerator class to generate synthetic regression data. We used make_regression for data creation, MinMaxScaler for scaling, and train_test_split for splitting into training, validation, and test sets. We have generated three different types of data: base dataset, larger dataset, and noisy dataset.
To generate new synthetic data

   ```bash
   cd ./data/generate_datasets.py
   python3 generate_datasets.py
   ```

This will generate data under directory `data/synthetic_data/` with the following files:
- `base_dataset.csv`
- `larger_dataset.csv`
- `noisy_dataset.csv`

### Running the tests
All tests can be found at the directory `src/supervised_ml/experiments`. We indicate how to run each experiment (we assume that the virtual environment is set to run any python code from here.)

#### 1 ANN derivative without Keras
Implementation is found at directory `models/ANN_derivative.py`.
Experiments for execution and analysis phases are separated in files `experiments/experiment_ANN_derivative.py` and `experiments/analysis_ANN_derivative.py` respectively.

To run the experiment file to generate test file run the following code:
   ```bash
   cd ./experiments/experiment_ANN_derivative.py
   python3 experiment_ANN_derivative.py
   ```

We will see the results file in `results/ANN_derivative/ann_derivative_test.csv`.

Then, to find the analysis run the following code
   ```bash
   python3 analysis_ANN_derivative.py
   ```


This will generate `results/ANN_derivative/best_ann_derivative_analyse.csv` and `results/ANN_derivative/worst_ann_derivative_analyse.csv`.
Where we can see best and worst parameters for the ANN.


#### 2 ANN derivative with Keras

Implementation is found at directory `models/ANN_derivative_keras.py`.
Experiments for execution and analysis phases are separated in files  `experiments/experiment_ANN_derivative_keras.py` and `experiments/analysis_ANN_derivative_keras.py` respectively.

To run the test file run the following code:
   ```bash
   cd ./experiments/experiment_ANN_derivative_keras.py
   python3 experiment_ANN_derivative_keras.py
   ```

We will see the results file in `results/ANN_derivative_keras/ann_derivative_keras_test.csv`.

Then, to run the analysis run the following code
   ```bash
   python3 analysis_ANN_derivative_keras.py
   ```


This will generate `results/ANN_derivative_keras/best_ann_derivative_keras_analyse.csv` and `results/ANN_derivative_keras/worst_ann_derivative_keras_analyse.csv`.
Where we can see best and worst parameters for the ANN using Keras.


#### 3 ANN Evolutionary Algorithm with GA

Implementation is found at directory `models/ANN_genetic.py`.
Experiments for execution and analysis phases are separated in files  `experiments/experiment_ANN_genetic_keras.py`.

To run the test file run the following code:
   ```bash
   cd ./experiments/experiment_ANN_genetic.py
   python3 experiment_ANN_genetic.py
   ```

We will see the results file in `results/ANN_genetic/ann_genetic_test.csv`.


#### 4 ANN Evolutionary Algorithm with GA

Implementation is found at directory `models/ANN_genetic_keras.py`.
Experiments for execution and analysis phases are separated in files  `experiments/experiment_ANN_genetic_keras.py` and `experiments/analysis_ANN_genetic_keras.py` respectively.

To run the test file run the following code:
   ```bash
   cd ./experiments/experiment_ANN_genetic_keras.py
   python3 experiment_ANN_genetic_keras.py
   ```

We will see the results file in `results/ANN_genetic_keras/ann_genetic_keras_test.csv`.

Then, to run the analysis run the following code
   ```bash
   python3 analysis_ANN_genetic_keras.py
   ```


This will generate `results/ANN_genetic_keras/best_ann_genetic_keras_analyse.csv` and `results/ANN_genetic_keras/worst_ann_genetic_keras_analyse.csv`.
Where we can see best and worst parameters for the ANN using Keras.

