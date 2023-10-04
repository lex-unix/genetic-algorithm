import random
import numpy as np
from numpy.random import randint
from genetic import Individ


def calculate_similarity(individ1: Individ, individ2: Individ) -> float:
    dx = individ1.x_phenotype - individ2.x_phenotype
    dy = individ1.y_phenotype - individ2.y_phenotype
    return np.sqrt(dx * dx + dy * dy)


def panmixia(population: list[Individ]):
    return [random.choice(population), random.choice(population)]


def selection(population: list[Individ]):
    mean = np.mean(np.round([individ.fitness for individ in population], 4))
    candidates = [individ for individ in population if individ.fitness <= mean]
    if not candidates:
        min_fitness = min([individ.fitness for individ in population])
        candidates = [
            individ for individ in population if individ.fitness == min_fitness
        ]
    return [random.choice(candidates), random.choice(candidates)]


def inbreeding(population: list[Individ]):
    min_similarity = float('inf')
    selected_parents = None

    for i in range(len(population)):
        for j in range(i+1, len(population)):
            similarity = calculate_similarity(population[i], population[j])
            if similarity < min_similarity:
                min_similarity = similarity
                selected_parents = [population[i], population[j]]

    return selected_parents


def outbreeding(population: list[Individ]):
    max_dissimilarity = -1
    selected_parents = None

    for i in range(len(population)):
        for j in range(i+1, len(population)):
            dissimilarity = calculate_similarity(
                population[i], population[j]
            )
            if dissimilarity > max_dissimilarity:
                max_dissimilarity = dissimilarity
                selected_parents = [population[i], population[j]]

    return selected_parents


def crossover(parent_a: Individ, parent_b: Individ) -> list[Individ]:
    point = randint(1, len(parent_a.genotype))

    chromo_a = parent_a.genotype[:point] + parent_b.genotype[point:]
    chromo_b = parent_b.genotype[:point] + parent_a.genotype[point:]

    child_a = Individ(genotype=chromo_a)
    child_b = Individ(genotype=chromo_b)

    return [child_a, child_b]


def two_point_crossover(parent_a: Individ, parent_b: Individ) -> list[Individ]:
    genotype_length = len(parent_a.genotype)

    cut_point_1, cut_point_2 = sorted(random.sample(range(genotype_length), 2))

    child_1_genotype = (
        parent_a.genotype[:cut_point_1] +
        parent_b.genotype[cut_point_1:cut_point_2] +
        parent_a.genotype[cut_point_2:]
    )

    child_2_genotype = (
        parent_b.genotype[:cut_point_1] +
        parent_a.genotype[cut_point_1:cut_point_2] +
        parent_b.genotype[cut_point_2:]
    )

    child_a = Individ(genotype=child_1_genotype)
    child_b = Individ(genotype=child_2_genotype)

    return [child_a, child_b]


def elitism(parents: list[Individ], children: list[Individ], num_elites: int = 10):
    combined = parents + children
    combined_sorted = sorted(combined, key=lambda x: x.fitness)
    return combined_sorted[:len(parents)]


def simple(children: list[Individ], _):
    return children


def select_with_displacement(parents: list[Individ], children: list[Individ], num_elites: int = 10):
    new_population = []
    for child in children:
        if child not in parents:
            new_population.append(child)
    if len(new_population) < len(parents):
        diff = len(parents) - len(new_population)
        sorted_parents = sorted(parents, key=lambda x: x.fitness)
        new_population.extend(sorted_parents[:diff])
    return new_population


def has_close_fitnesses(parents: list[Individ], children: list[Individ], e: float):
    parents_mean = np.mean([individ.fitness for individ in parents])
    children_mean = np.mean([individ.fitness for individ in children])
    diff = np.abs(parents_mean - children_mean)
    return diff < e


def has_close_solutions(children: list[Individ], parents, e: float,):
    n = len(children)
    min_distance = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            distance = calculate_similarity(children[i], children[j])
            if distance < min_distance:
                min_distance = distance

    return min_distance < e
