import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygad.gann
import pygad.nn
import pygad

from sklearn.model_selection import train_test_split

from utils_models import fitness_regression, callback_generation_default
from utils_metrics import metrics_report_regression, save_results_to_csv

import warnings
warnings.filterwarnings("ignore")


class ANN_Genetic:
    def __init__(self, train, target, val_train, val_target, arguments_GANN, arguments_GA, problem="regression"):
        """

        :param train:
        :param target:
        :param val_train:
        :param val_target:
        :param arguments_GANN:
        :param arguments_GA:
        :param problem:
        """
        # Prepare training data
        self.train = np.asarray(train, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        self.val_train = np.asarray(val_train, dtype=np.float32)
        self.val_target = np.asarray(val_target, dtype=np.float32)

        num_neurons_input = train.shape[1]
        self.last_fitness = [0]
        self.problem = problem

        # GANN instance to train NN using GA
        self.gann = pygad.gann.GANN( num_solutions = arguments_GANN["num_solutions"], # Number of neural networks (solutions) in the population.
                                     num_neurons_input = num_neurons_input, # Number of neurons in the input layer
                                     num_neurons_hidden_layers = arguments_GANN["num_neurons_hidden_layers"], # List holding the number of neurons in the hidden layers
                                     num_neurons_output = arguments_GANN["num_neurons_output"], # Number of neurons in the output layer
                                     hidden_activations = arguments_GANN["hidden_activations"], # List holding the names of the activation functions of the hidden layers
                                     output_activation = arguments_GANN["output_activation"]) # Name of the activation function of the output layer

        self.population_vectors = pygad.gann.population_as_vectors(population_networks=self.gann.population_networks)

        # Building a genetic algorithm
        self.ga = pygad.GA(num_generations = arguments_GA["num_generations"], # Number of generations.
                           num_parents_mating = arguments_GA["num_parents_mating"], # Number of solutions to be selected as parents.
                           initial_population = self.population_vectors.copy(), # A user-defined initial population.
                           fitness_func = arguments_GA["fitness_func"](self.gann, self.train, self.target), # Accepts a function/method and returns the fitness value(s) of the solution.
                           mutation_percent_genes = arguments_GA["mutation_percent_genes"], # Percentage of genes to mutate. It defaults to the string "default",  10%
                           init_range_low = arguments_GA["init_range_low"], # The lower value of the random range from which the gene values in the initial population are selected.
                           init_range_high = arguments_GA["init_range_high"], # The upper value of the random range from which the gene values in the initial population are selected.
                           parent_selection_type = arguments_GA["parent_selection_type"], # The parent selection type: "sss", "rws", "sus", "rank", "random" and "tournament"
                           crossover_type = arguments_GA["crossover_type"], # Type of the crossover operation: "single_point", "two_points", "uniform" and "scattered"
                           mutation_type = arguments_GA["mutation_type"], # Type of the mutation operation: "random", "swap", "inversion","scramble" and "adaptive"
                           keep_parents = arguments_GA["keep_parents"], # Number of parents to keep in the current population.
                           on_generation = arguments_GA["on_generation"](self.gann, self.last_fitness), # Accepts a function to be called after each generation.
                           save_solutions = True)

        # Save parameters
        self.arguments_GANN = arguments_GANN
        self.arguments_GA = arguments_GA

        # Train results
        self.solution = None
        self.solution_fitness = None
        self.solution_idx = None


    def fit(self):
        self.ga.run()

        # Save best solution
        self.solution, self.solution_fitness, self.solution_idx = self.ga.best_solution(pop_fitness=self.ga.last_generation_fitness)

    def predict(self, test):
        return pygad.nn.predict(last_layer = self.gann.population_networks[self.solution_idx],
                                data_inputs = np.asarray(test, dtype=np.float32),
                                problem_type = self.problem)

    def ga_plots(self):
        self.ga.plot_fitness()
        self.ga.plot_genes()
        self.ga.plot_new_solution_rate()

        print("Parameters of the best solution : {solution}".format(solution = self.solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness = self.solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx = self.solution_idx))

        if self.ga.best_solution_generation != -1:
            print("Best fitness value reached after {best_solution_generation} generations.".format(
                best_solution_generation=self.ga.best_solution_generation))



# Testing implementation
if __name__ == "__main__":

    # Testing testing 1 2 3
    data = np.random.normal(0, 1, (500,5))
    dataset = pd.DataFrame(data, columns = ["f1", "f2", "f3", "f4", "target"])

    x_train, x_test, y_train, y_test = train_test_split(dataset[["f1", "f2", "f3", "f4"]], dataset["target"], test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.5, random_state=1)

    print(x_train.shape)
    print(y_train.shape)

    # Parameters
    parameters_GANN = { "num_solutions": 6,
                        "num_neurons_input": 4,
                        "num_neurons_hidden_layers": [2],
                        "num_neurons_output": 1,
                        "hidden_activations": ["relu"],
                        "output_activation": "None" }

    parameters_GA = { "num_generations": 50,
                      "num_parents_mating": 4,
                      "fitness_func": fitness_regression,
                      "mutation_percent_genes": 5,
                      "init_range_low": -1,
                      "init_range_high": 1,
                      "parent_selection_type": "sss",
                      "crossover_type": "single_point",
                      "mutation_type": "random",
                      "keep_parents": 1,
                      "on_generation": callback_generation_default }

    # Genetic ANN
    gann = ANN_Genetic(x_train, y_train, x_val, y_val, parameters_GANN, parameters_GA)

    # Training
    start_train_time = time.time ()
    gann.fit()
    gann.ga_plots()
    end_train_time = time.time ()
    training_time = end_train_time - start_train_time
    # Testing
    start_test_time = time.time ()
    predictions = gann.predict(x_test)
    predictions = np.asarray(predictions)
    end_test_time = time.time ()

    testing_time = end_test_time - start_test_time

    print(y_test.shape)
    print(predictions.shape)
    mae, mse, max_err, r2 = metrics_report_regression (y_test, predictions)
    results = []
    print(mae, mse, max_err, r2)
    results.append ({
        "test_time": testing_time,
        "training_time": training_time,
        "mse": mse,
        "mae": mae,
        "max_error": max_err,
        "r2": r2
    })
    DATA_SET_PATH = Path (__file__).parent.parent.joinpath ("datasets").resolve ()
    results_file_path = DATA_SET_PATH.joinpath ("Ann_genetics_results.csv")
    save_results_to_csv (results, results_file_path)