import sys
import options
from genetic import GeneticAlgorithm
from functions import Easom, ThreeHumpCamel, Ackley


def test_mutation(func):
    rates = [0.001, 0.05, 0.01]
    for rate in rates:
        algo = GeneticAlgorithm(epochs=50, size=300)

        algo.set_mutation_rate(rate)
        algo.set_inversion_rate(0.05)

        algo.set_crossover(options.crossover, 0.75)
        algo.set_parent_selection(options.selection)
        algo.set_create_populationa(options.elitism)
        result = algo.solve(func)
        x, y, f = result.x_phenotype, result.y_phenotype, result.fitness
        func_name = type(func).__name__
        print(f'rate {rate}: ${func_name} {x=:.4f} {y=:.4f} {f=:.4f}')


def test_crossover(func):
    crossovers = [options.crossover, options.two_point_crossover]
    for crossover in crossovers:
        algo = GeneticAlgorithm(epochs=50, size=300)

        algo.set_mutation_rate(0.075)
        algo.set_inversion_rate(0.05)

        algo.set_crossover(crossover, 0.75)
        algo.set_parent_selection(options.selection)
        algo.set_create_populationa(options.elitism)
        result = algo.solve(func)
        x, y, f = result.x_phenotype, result.y_phenotype, result.fitness
        func_name = type(func).__name__
        print(
            f'method {crossover.__name__}: ${func_name} {x=:.4f} {y=:.4f} {f=:.4f}')


def test_parents(func):
    methods = [options.panmixia, options.selection,
               options.inbreeding, options.outbreeding]
    for method in methods:
        algo = GeneticAlgorithm(epochs=50, size=300)

        algo.set_mutation_rate(0.075)
        algo.set_inversion_rate(0.05)

        algo.set_crossover(options.crossover, 0.75)
        algo.set_parent_selection(method)
        algo.set_create_populationa(options.elitism)
        result = algo.solve(func)
        x, y, f = result.x_phenotype, result.y_phenotype, result.fitness
        func_name = type(func).__name__
        print(
            f'method {method.__name__}: ${func_name} {x=:.4f} {y=:.4f} {f=:.4f}')


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
        result = algo.solve(func)
        x, y, f = result.x_phenotype, result.y_phenotype, result.fitness
        func_name = type(func).__name__
        print(
            f'method {method.__name__}: ${func_name} {x=:.4f} {y=:.4f} {f=:.4f}')


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


if __name__ == '__main__':
    main()
