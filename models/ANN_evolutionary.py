
import numpy as np
import pandas as pd
import pygad.gann
import pygad.nn
import pygad
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


class ANN_Evolutionary:
    def __init__(self, train, target, arguments_GANN, arguments_GA):
        """

        :param train:
        :param target:
        :param arguments_GANN:
        :param arguments_GA:
        """
        # Prepare training data
        self.train = np.asarray(train, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

        num_neurons_input = train.shape[1]

        # Initialize GANN
        self.gann = pygad.gann.GANN(num_solutions = arguments_GANN["num_solutions"],
                                    num_neurons_input = num_neurons_input,
                                    num_neurons_hidden_layers = arguments_GANN["num_neurons_hidden_layers"],
                                    num_neurons_output = arguments_GANN["num_neurons_output"],
                                    hidden_activations = arguments_GANN["hidden_activations"],
                                    output_activation = arguments_GANN["output_activation"])

        self.population_vectors = pygad.gann.population_as_vectors(population_networks=self.gann.population_networks)
        self.ga = pygad.GA(num_generations = arguments_GA["num_generations"],
                           num_parents_mating = arguments_GA["num_parents_mating"],

                           #
                           initial_population = self.population_vectors.copy(),
                           fitness_func = arguments_GA["fitness_func"],
                           mutation_percent_genes = arguments_GA["mutation_percent_genes"],
                           init_range_low = arguments_GA["init_range_low"],
                           init_range_high = arguments_GA["init_range_high"],
                           parent_selection_type = arguments_GA["parent_selection_type"],
                           crossover_type = arguments_GA["crossover_type"],
                           mutation_type = arguments_GA["mutation_type"],
                           keep_parents = arguments_GA["keep_parents"],
                           on_generation = arguments_GA["on_generation"])

        # Save parameters
        self.arguments_GANN = arguments_GANN
        self.arguments_GA = arguments_GA

        # Train results
        self.solution = None
        self.solution_fitness = None
        self.solution_idx = None

    def fit(self, plot=False):
        self.ga.run()

        if plot:
            self.ga.plot_fitness()

        self.solution, self.solution_fitness, self.solution_idx = self.ga.best_solution(pop_fitness=self.ga.last_generation_fitness)

    def predict(self, test):
        return pygad.nn.predict(last_layer = self.gann.population_networks[self.solution_idx],
                                data_inputs = np.asarray(test, dtype=np.float32),
                                problem_type = "regression")


def fitness_func(ga_instance, solution, sol_idx):
    global GANN_instance, data_inputs, data_outputs

    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                   data_inputs=data_inputs, problem_type="regression")
    solution_fitness = 1.0 / np.mean(np.abs(predictions - data_outputs))

    return solution_fitness


def callback_generation(ga_instance):
    global GANN_instance, last_fitness

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                            population_vectors=ga_instance.population)

    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    print("Change     = {change}".format(
        change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))

    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1].copy()


if __name__ == "__main__":

    # Testing testing 1 2 3
    data = np.random.normal(0, 1, (500,5))
    dataset = pd.DataFrame(data, columns = ["f1", "f2", "f3", "f4", "target"])

    x_train, x_test, y_train, y_test = train_test_split(dataset[["f1", "f2", "f3", "f4"]],
                                                        dataset["target"], test_size=0.1, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)

    print(x_train.shape)
    print(y_train.shape)


    # Parameters
    parameters_GANN = { "num_solutions": 6,
                        "num_neurons_input": 4,
                        "num_neurons_hidden_layers": [2],
                        "num_neurons_output": 1,
                        "hidden_activations": ["relu"],
                        "output_activation": "None" }

    parameters_GA = { "num_generations": 500,
                      "num_parents_mating": 4,
                      "fitness_func": fitness_func,
                      "mutation_percent_genes": 5,
                      "init_range_low": -1,
                      "init_range_high": 1,
                      "parent_selection_type": "sss",
                      "crossover_type": "single_point",
                      "mutation_type": "random",
                      "keep_parents": 1,
                      "on_generation": callback_generation }

    # Global parameters
    last_fitness = 0

    # Genetic ANN
    gann = ANN_Evolutionary(x_train, y_train, parameters_GANN, parameters_GA)

    # Global parameters
    GANN_instance = gann.gann
    data_inputs = gann.train
    data_outputs = gann.target

    # Training
    gann.fit(True)

    # Testing
    predictions = gann.predict(x_test)
    print("Predictions of the trained network : {predictions}".format(predictions=predictions))

    # Calculating some statistics
    abs_error = np.mean(np.abs(predictions - np.asarray(y_test, dtype=np.float32)))
    print("Absolute error : {abs_error}.".format(abs_error=abs_error))


    """
    https://pygad.readthedocs.io/en/latest/
    # Holds the fitness value of the previous generation.
    last_fitness = 0

    data_train = np.array(x_train)
    label_train = np.array(y_train)


    # Preparing the NumPy array of the inputs.
    data_inputs = np.asarray(data_train, dtype=np.float32)

    # Preparing the NumPy array of the outputs.
    data_outputs = np.asarray(label_train, dtype=np.float32)

    print(data_inputs.shape)
    print(data_outputs.shape)

    # The length of the input vector for each sample (i.e. number of neurons in the input layer).
    num_inputs = data_inputs.shape[1]

    # Creating an initial population of neural networks. The return of the initial_population() function holds references to the networks, not their weights. Using such references, the weights of all networks can be fetched.
    num_solutions = 6  # A solution or a network can be used interchangeably.
    GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                    num_neurons_input=num_inputs,
                                    num_neurons_hidden_layers=[2],
                                    num_neurons_output=1,
                                    hidden_activations=["relu"],
                                    output_activation="None")

    # population does not hold the numerical weights of the network instead it holds a list of references to each last layer of each network (i.e. solution) in the population. A solution or a network can be used interchangeably.
    # If there is a population with 3 solutions (i.e. networks), then the population is a list with 3 elements. Each element is a reference to the last layer of each network. Using such a reference, all details of the network can be accessed.
    population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

    # To prepare the initial population, there are 2 ways:
    # 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
    # 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
    initial_population = population_vectors.copy()

    num_parents_mating = 4  # Number of solutions to be selected as parents in the mating pool.
    num_generations = 500  # Number of generations.
    mutation_percent_genes = 5  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
    parent_selection_type = "sss"  # Type of parent selection.
    crossover_type = "single_point"  # Type of the crossover operator.
    mutation_type = "random"  # Type of the mutation operator.
    keep_parents = 1  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

    init_range_low = -1
    init_range_high = 1

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           mutation_percent_genes=mutation_percent_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           keep_parents=keep_parents,
                           on_generation=callback_generation)

    ga_instance.run()

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_fitness()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    if ga_instance.best_solution_generation != -1:
        print("Best fitness value reached after {best_solution_generation} generations.".format(
            best_solution_generation=ga_instance.best_solution_generation))

    # Predicting the outputs of the data using the best solution.
    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[solution_idx],
                                   data_inputs=data_inputs,
                                   problem_type="regression")
    print("Predictions of the trained network : {predictions}".format(predictions=predictions))

    # Calculating some statistics
    abs_error = np.mean(np.abs(predictions - data_outputs))
    print("Absolute error : {abs_error}.".format(abs_error=abs_error))


    # https://github.com/ahmedfgad/NeuralGenetic/blob/master/Fish.csv
    # https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad/
    """

    # XOR implementation (not ANN?)
    """
    # Holds the fitness value of the previous generation.
    last_fitness = 0

    # Preparing the NumPy array of the outputs.
    data_inputs = np.array([ [2, 5, -3, 0.1], [8, 15, 20, 13] ])
    data_outputs = np.array([ [0.1, 0.2],[1.8, 1.5] ])

    print(data_inputs.shape)
    print(data_outputs.shape)


    # The length of the input vector for each sample (i.e. number of neurons in the input layer).
    num_inputs = data_inputs.shape[1]
    

    # Creating an initial population of neural networks. The return of the initial_population() function holds references to the networks, not their weights. Using such references, the weights of all networks can be fetched.
    num_solutions = 6  # A solution or a network can be used interchangeably.
    GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                    num_neurons_input=num_inputs,
                                    num_neurons_hidden_layers=[2],
                                    num_neurons_output=2,
                                    hidden_activations=["relu"],
                                    output_activation="None")
                                    
    # population does not hold the numerical weights of the network instead it holds a list of references to each last layer of each network (i.e. solution) in the population. A solution or a network can be used interchangeably.
    # If there is a population with 3 solutions (i.e. networks), then the population is a list with 3 elements. Each element is a reference to the last layer of each network. Using such a reference, all details of the network can be accessed.
    population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

    # To prepare the initial population, there are 2 ways:
    # 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
    # 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
    initial_population = population_vectors.copy()

    num_parents_mating = 4  # Number of solutions to be selected as parents in the mating pool.

    num_generations = 500  # Number of generations.

    mutation_percent_genes = 5  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.

    parent_selection_type = "sss"  # Type of parent selection.

    crossover_type = "single_point"  # Type of the crossover operator.

    mutation_type = "random"  # Type of the mutation operator.

    keep_parents = 1  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

    init_range_low = -1
    init_range_high = 1

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           mutation_percent_genes=mutation_percent_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           keep_parents=keep_parents,
                           on_generation=callback_generation)

    ga_instance.run()

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_fitness()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    if ga_instance.best_solution_generation != -1:
        print("Best fitness value reached after {best_solution_generation} generations.".format(
            best_solution_generation=ga_instance.best_solution_generation))

    # Predicting the outputs of the data using the best solution.
    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[solution_idx],
                                   data_inputs=data_inputs,
                                   problem_type="regression")
    print("Predictions of the trained network : {predictions}".format(predictions=predictions))

    # Calculating some statistics
    abs_error = np.mean(np.abs(predictions - data_outputs))
    print("Absolute error : {abs_error}.".format(abs_error=abs_error))
    """