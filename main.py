import time
import sys
import options
from genetic import GeneticAlgorithm
from functions import Easom, ThreeHumpCamel, Ackley


def start_timer():
    start = time.time()

    def stop_timer():
        print(f'Time to finish: {time.time() - start:.2f} seconds\n')

    return stop_timer


def test_mutation(func):
    rates = [0.001, 0.05, 0.01]
    for rate in rates:
        algo = GeneticAlgorithm(epochs=50, size=300)

        algo.set_mutation_rate(rate)
        algo.set_inversion_rate(0.05)
        algo.set_crossover(options.crossover, 0.75)
        algo.set_parent_selection(options.selection)
        algo.set_create_populationa(options.elitism)

        stop_timer = start_timer()
        result = algo.solve(func)

        x, y, f = result.x_phenotype, result.y_phenotype, result.fitness
        func_name = type(func).__name__
        print(
            f'rate {rate}: {func_name} {x=:.4f} {y=:.4f} {f=:.4f}')
        stop_timer()


def test_crossover(func):
    crossovers = [options.crossover, options.two_point_crossover]
    for crossover in crossovers:
        algo = GeneticAlgorithm(epochs=50, size=300)

        algo.set_mutation_rate(0.075)
        algo.set_inversion_rate(0.05)
        algo.set_crossover(crossover, 0.75)
        algo.set_parent_selection(options.selection)
        algo.set_create_populationa(options.elitism)

        stop_timer = start_timer()
        result = algo.solve(func)
        x, y, f = result.x_phenotype, result.y_phenotype, result.fitness
        func_name = type(func).__name__
        print(
            f'method {crossover.__name__}: {func_name} {x=:.4f} {y=:.4f} {f=:.4f}')
        stop_timer()


def test_parents(func):
    methods = [options.panmixia, options.selection,
               options.inbreeding, options.outbreeding]
    for method in methods:
        algo = GeneticAlgorithm(epochs=20, size=200)

        algo.set_mutation_rate(0.075)
        algo.set_inversion_rate(0.05)
        algo.set_crossover(options.crossover, 0.75)
        algo.set_parent_selection(method)
        algo.set_create_populationa(options.elitism)

        stop_timer = start_timer()

        result = algo.solve(func)
        x, y, f = result.x_phenotype, result.y_phenotype, result.fitness
        func_name = type(func).__name__
        print(
            f'method {method.__name__}: {func_name} {x=:.4f} {y=:.4f} {f=:.4f}')
        stop_timer()


def test_population_creation(func):
    methods = [options.simple, options.elitism,
               options.select_with_displacement]
    for method in methods:
        algo = GeneticAlgorithm(epochs=50, size=300)

        algo.set_mutation_rate(0.075)
        algo.set_inversion_rate(0.05)

        algo.set_crossover(options.crossover, 0.75)
        algo.set_parent_selection(options.selection)
        algo.set_create_populationa(method)
        stop_timer = start_timer()
        result = algo.solve(func)
        x, y, f = result.x_phenotype, result.y_phenotype, result.fitness
        func_name = type(func).__name__
        print(
            f'method {method.__name__}: {func_name} {x=:.4f} {y=:.4f} {f=:.4f}')
        stop_timer()


def test_stop(func):
    methods = [None, options.has_close_fitnesses, options.has_close_solutions]
    for method in methods:
        epochs = 50
        algo = GeneticAlgorithm(
            epochs=epochs if method is None else 1_000_000, size=300
        )

        algo.set_mutation_rate(0.075)
        algo.set_inversion_rate(0.05)
        algo.set_crossover(options.crossover, 0.75)
        algo.set_parent_selection(options.selection)
        algo.set_create_populationa(options.elitism)
        algo.set_stop_criteria(method, 0.01)

        stop_timer = start_timer()
        result = algo.solve(func)
        if isinstance(result, list):
            result, epochs = result
        x, y, f = result.x_phenotype, result.y_phenotype, result.fitness
        func_name = type(func).__name__
        method_name = method.__name__ if method is not None else 'iterations'
        print(
            f'method {method_name}: {func_name} {x=:.4f} {y=:.4f} {f=:.4f} for {epochs} epochs')
        stop_timer()


def main():
    objective = sys.argv[1]
    funcs = [Ackley, Easom, ThreeHumpCamel]
    if objective == 'mutation':
        for func in funcs:
            test_mutation(func())
    if objective == 'parents':
        for func in funcs:
            test_parents(func())
    if objective == 'crossover':
        for func in funcs:
            test_crossover(func())
    if objective == 'population':
        for func in funcs:
            test_population_creation(func())
    if objective == 'stop':
        for func in funcs:
            test_stop(func())


if __name__ == '__main__':
    main()
