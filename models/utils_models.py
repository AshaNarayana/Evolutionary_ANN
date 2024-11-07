
import numpy as np
import pygad.gann
import pygad.nn
import pygad



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
