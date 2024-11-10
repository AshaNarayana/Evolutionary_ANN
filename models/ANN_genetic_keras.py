
import tensorflow.keras # type: ignore
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

import pygad.kerasga
import pygad

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from utils_models import fitness_regression_keras, callback_generation_keras
from utils_metrics import regression_report

import warnings
warnings.filterwarnings("ignore")


class ANN_Genetic_keras:
    def __init__(self, train, target, val_train, val_target, arguments_GA, problem="regression"):
        """
        Initialize neural networks + Keras model
        :param train: Train dataset
        :param target: Target values from training set
        :param val_train: Validation dataset
        :param val_target: Target values from validation set
        :param arguments_GA: Dictionary with arguments
        :param problem: Set as regression
        """
        # Prepare training data
        self.train = np.asarray(train, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        self.val_train = np.asarray(val_train, dtype=np.float32)
        self.val_target = np.asarray(val_target, dtype=np.float32)

        # Create model (using best derivative architecture)
        input_layer  = tensorflow.keras.layers.Input((train.shape[1],))
        dense_layer1 = tensorflow.keras.layers.Dense(150, activation="sigmoid")(input_layer)
        output_layer = tensorflow.keras.layers.Dense(target.shape[1], activation="linear")(dense_layer1)
        self.model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

        self.model.compile(optimizer=Adam(learning_rate=0.01),
                           loss='mean_squared_error',
                           metrics=[MeanSquaredError(), MeanAbsoluteError()] )

        self.weights_vector = pygad.kerasga.model_weights_as_vector(model=self.model)
        self.keras_ga = pygad.kerasga.KerasGA(model=self.model, num_solutions=10) # Num solutions
        
        # Building a genetic algorithm
        self.ga = pygad.GA(num_generations = arguments_GA["num_generations"], # Number of generations.
                           num_parents_mating = arguments_GA["num_parents_mating"], # Number of solutions to be selected as parents.

                           # Set initial population manually
                           initial_population = self.keras_ga.population_weights, # A user-defined initial population.

                           fitness_func = arguments_GA["fitness_func"](self.train, self.target, self.keras_ga, self.model), # Accepts a function/method and returns the fitness value(s) of the solution.
                           mutation_percent_genes=arguments_GA["mutation_percent_genes"], # Percentage of genes to mutate. It defaults to the string "default",  10%
                           init_range_low=arguments_GA["init_range_low"], # The lower value of the random range from which the gene values in the initial population are selected.
                           init_range_high=arguments_GA["init_range_high"], # The upper value of the random range from which the gene values in the initial population are selected.
                           parent_selection_type=arguments_GA["parent_selection_type"], # The parent selection type: "sss", "rws", "sus", "rank", "random" and "tournament"
                           crossover_type=arguments_GA["crossover_type"], # Type of the crossover operation: "single_point", "two_points", "uniform" and "scattered"
                           mutation_type=arguments_GA["mutation_type"], # Type of the mutation operation: "random", "swap", "inversion","scramble" and "adaptive"
                           keep_parents=arguments_GA["keep_parents"], # Number of parents to keep in the current population.
                           on_generation = arguments_GA["on_generation"], # Accepts a function to be called after each generation.
                           save_solutions = True)

        # Save parameters
        self.arguments_GA = arguments_GA

        # Train past_results
        self.solution = None
        self.solution_fitness = None
        self.solution_idx = None


    def fit(self):
        self.ga.run()
        
        # Save best solution
        self.solution, self.solution_fitness, self.solution_idx = self.ga.best_solution(pop_fitness=self.ga.last_generation_fitness)

        # Fetch the parameters of the best solution.
        best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=self.model, weights_vector=self.solution)
        self.model.set_weights(best_solution_weights)


    def predict(self, test):
        """
        Make predictions
        :param test: Testing data set
        :return: Predictions
        """
        return self.model.predict(test)
    

    def ga_plots(self):
        self.ga.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)
        #self.ga.plot_genes()
        #self.ga.plot_new_solution_rate()

        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=self.solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=self.solution_idx))




# Testing implementation
if __name__ == "__main__":

    # Testing testing 1 2 3
    data = np.random.normal(0, 1, (500,5))
    dataset = pd.DataFrame(data, columns = ["f1", "f2", "f3", "f4", "target"])

    x_train, x_test, y_train, y_test = train_test_split(dataset[["f1", "f2", "f3", "f4"]], dataset["target"], test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.5, random_state=1)

    print(x_train.shape)
    print(y_train.shape)

    parameters_GA = { "num_generations": 15,
                      "num_parents_mating": 4,
                      "fitness_func": fitness_regression_keras,
                      "mutation_percent_genes": 5,
                      "init_range_low": -1,
                      "init_range_high": 1,
                      "parent_selection_type": "sss",
                      "crossover_type": "single_point",
                      "mutation_type": "random",
                      "keep_parents": 1,
                      "on_generation": callback_generation_keras }
    
    # Genetic ANN
    gann_keras = ANN_Genetic_keras(x_train, y_train, x_val, y_val, parameters_GA)

    # Training
    gann_keras.fit()
    gann_keras.ga_plots()

    # Testing
    predictions = gann_keras.predict(x_test)

    # Results
    print(y_test.shape)
    print(predictions.shape)

    print(regression_report(y_test, predictions))