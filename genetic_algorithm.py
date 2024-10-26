import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size, gene_length, gene_elements, fitness_func, mutation_rate=0.01, crossover_rate=0.7, generations=100):
        """
        :param pop_size: 集団のサイズ（個体数）
        :param gene_length: 遺伝子の長さ（各個体の構成要素の数）
        :param gene_elements: 遺伝子要素のリスト（例：["up", "down", "left", "right"]）
        :param fitness_func: 適応度関数（評価関数）
        :param mutation_rate: 突然変異率
        :param crossover_rate: 交叉率
        :param generations: 実行する世代数
        """
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.gene_elements = gene_elements
        self.fitness_func = fitness_func
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population = self.initialize_population()
        
    def initialize_population(self):
        # 個体集団の初期化（ランダムにgene_elementsから選択）
        return np.random.choice(self.gene_elements, size=(self.pop_size, self.gene_length))
    
    def calculate_fitness(self):
        # 適応度を計算する
        fitness_values = np.array([self.fitness_func(ind) for ind in self.population])
        return fitness_values
    
    def select_parents(self, fitness_values):
        # 適応度に基づき親個体を選択する（ルーレット選択）
        total_fitness = np.sum(fitness_values)
        selection_probs = fitness_values / total_fitness
        parents_indices = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=selection_probs)
        return self.population[parents_indices]
    
    def crossover(self, parents):
        # 交叉を行い、新しい子孫を生成する
        offspring = []
        for i in range(0, self.pop_size, 2):
            parent1, parent2 = parents[i], parents[(i+1) % self.pop_size]
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.gene_length)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parent1)
                offspring.append(parent2)
        return np.array(offspring)
    
    def mutate(self, offspring):
        # 突然変異を行い、解の多様性を保持する
        for individual in offspring:
            for gene in range(self.gene_length):
                if np.random.rand() < self.mutation_rate:
                    # gene_elementsのいずれかの要素で置換
                    individual[gene] = np.random.choice(self.gene_elements)
        return offspring