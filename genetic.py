import struct
from numpy.random import rand
from numpy.random import randint
from numpy.random import uniform
import math


class Individ:
    bounds = []

    def __init__(self, x: float = None, y: float = None, genotype: str = None):
        self.fitness = None
        if genotype is None:
            self.x_phenotype = x
            self.y_phenotype = y
            self.genotype = self.to_genotype(x) + self.to_genotype(y)
        else:
            mid = len(genotype) // 2
            binary_x, binary_y = genotype[:mid], genotype[mid:]
            x = self.to_phenotype(binary_x)
            y = self.to_phenotype(binary_y)
            if math.isnan(x) or math.isnan(y):
                self.recover()
            else:
                self.genotype = genotype
                self.x_phenotype = x
                self.y_phenotype = y

    def recover(self):
        bounds_x, bounds_y = self.bounds
        x = uniform(bounds_x[0], bounds_x[1])
        y = uniform(bounds_y[0], bounds_y[1])
        self.x_phenotype = x
        self.y_phenotype = y
        self.genotype = self.to_genotype(x) + self.to_genotype(y)

    def to_genotype(self, value):
        s = struct.pack('!f', value)
        return ''.join(format(c, '08b') for c in s)

    def to_phenotype(self, binary: str):
        bytes_val = int(binary, 2).to_bytes(4, byteorder='big')
        float_val = struct.unpack('!f', bytes_val)[0]
        return float_val

    def __eq__(self, other):
        if isinstance(other, Individ):
            return self.genotype == other.genotype and self.fitness == other.fitness
        return False

    @classmethod
    def set_bounds(cls, bounds):
        cls.bounds = bounds

    @classmethod
    def set_fitness(cls, fitness_func):
        cls.fitness_func = fitness_func


def create_population(bounds, size) -> list[Individ]:
    population = []
    for n in range(size):
        x_bounds = bounds[0]
        y_bounds = bounds[1]
        individ = Individ(
            uniform(x_bounds[0], x_bounds[1] + 1),
            uniform(y_bounds[0], y_bounds[1] + 1)
        )
        population.append(individ)
    return population


class GeneticAlgorithm:
    def __init__(self, size=100, epochs=5000):
        self.size = size
        self.epochs = epochs
        self.n_bits = 16

    def set_create_populationa(self, create_population):
        self.create_population = create_population

    def set_parent_selection(self, parent_selection):
        self.parent_selection = parent_selection

    def set_crossover(self, crossover, rate):
        self.crossover = crossover
        self.crossover_rate = rate

    def set_mutation_rate(self, rate):
        self.mutation_rate = rate

    def set_inversion_rate(self, rate):
        self.inversion_rate = rate

    def mutation(self, individ: Individ):
        i = randint(0, len(individ.genotype))
        bit = '1' if individ.genotype[i] == '0' else '1'
        individ.genotype = individ.genotype[:i] + bit + individ.genotype[i+1:]

    def inversion(self, individ: Individ):
        point = randint(1, len(individ.genotype))
        individ.genotype = individ.genotype[point:] + individ.genotype[:point]

    def solve(self, objective):
        bounds = objective.bounds
        func = objective.func
        Individ.set_bounds(bounds)
        Individ.set_fitness(func)

        population = create_population(bounds, self.size)

        for epoch in range(self.epochs):
            for individ in population:
                individ.fitness = func(
                    individ.x_phenotype, individ.y_phenotype
                )

            new_population = []
            for individ in population:
                parent_a, parent_b = self.parent_selection(population)
                new_individ = parent_a if 0.5 < rand() else parent_b

                if self.crossover_rate > rand():
                    child_a, child_b = self.crossover(parent_a, parent_b)
                    new_individ = child_a if 0.5 < rand() else child_b

                if self.mutation_rate > rand():
                    self.mutation(new_individ)

                if self.inversion_rate > rand():
                    self.inversion(new_individ)

                new_individ.fitness = func(
                    new_individ.x_phenotype, new_individ.y_phenotype)

                new_population.append(new_individ)

            population = self.create_population(population, new_population)

        return sorted(population, key=lambda x: x.fitness)[0]
