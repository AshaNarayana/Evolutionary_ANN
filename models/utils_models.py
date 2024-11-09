
import numpy as np
import tensorflow.keras # type: ignore
import pygad.kerasga
import pygad.gann
import pygad.nn
import pygad


def default_parameters_ga():
    parameters_GA = {"num_generations": 250,
                     "num_parents_mating": 4,
                     "fitness_func": fitness_regression_keras,
                     "mutation_percent_genes": "default",
                     "init_range_low": -4,
                     "init_range_high": 4,
                     "parent_selection_type": "sss",
                     "crossover_type": "single_point",
                     "mutation_type": "random",
                     "keep_parents": -1,
                     "on_generation": callback_generation_keras}
    return  parameters_GA



"""
Manual implementation
"""

def fitness_regression(gann, inputs, outputs):
    """

    :param gann:
    :param inputs:
    :param outputs:
    :return:
    """
    def fitness_func(ga_instance, solution, sol_idx):
        predictions = pygad.nn.predict(last_layer=gann.population_networks[sol_idx],
                                       data_inputs=inputs, problem_type="regression")
        return 1.0 / np.mean(np.abs(predictions - outputs))
    return fitness_func

def callback_generation_default(gann, last_fitness):
    """

    :param gann:
    :param last_fitness:
    :return:
    """
    def callback_generation(ga_instance):
        population_matrices = pygad.gann.population_as_matrices(population_networks=gann.population_networks,
                                                                population_vectors=ga_instance.population)
        gann.update_population_trained_weights(population_trained_weights=population_matrices)

        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
        print("Change     = {change}\n".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness[-1]))

        last_fitness.append(ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1].copy())

    return callback_generation



"""
Keras implementation
"""

def fitness_regression_keras(inputs, outputs, keras_ga, model):
    """

    :param inputs:
    :param outputs:
    :param keras_ga:
    :param model:
    :return:
    """
    def fitness_func(ga_instance, solution, sol_idx):
        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
        model.set_weights(weights=model_weights_matrix)
        #predictions = pygad.kerasga.predict(model=model,solution=solution,data=data_inputs)
        predictions = model.predict(inputs)

        mae = tensorflow.keras.losses.MeanAbsoluteError()
        abs_error = mae(outputs, predictions).numpy() + 0.00000001
        solution_fitness = 1.0 / abs_error

        return solution_fitness
    return fitness_func

def callback_generation_keras(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))