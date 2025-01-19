# Importing necessary libraries and modules
import itertools  # Provides tools for iterating over data in various ways, such as creating combinations or permutations.
import matplotlib.pyplot as plt  # Used for creating visualizations and plots.
from datetime import datetime  # Handles date and time data.
import json  # Used for reading and writing JSON files.
import pandas as pd  # Provides data manipulation and analysis tools.
from typing import Dict, List, Any, Tuple  # Specifies types for function arguments and return values.
import seaborn as sns  # Enhances data visualization capabilities with advanced plots.
import time  # Provides functions for tracking time and measuring durations.
from pathlib import Path  # Handles file system paths in a platform-independent way.
import random  # Generates random numbers and performs random operations.
from snake_game import SnakeGame, Snake, Food, Vector, get_input_state  # Custom imports for the Snake game and its related classes/functions.
from snake_ai_recording import SimpleModel, evaluate_fitness  # Custom imports for AI model and fitness evaluation.
import numpy as np  # Provides numerical computing tools for arrays, matrices, and mathematical operations.
import matplotlib  # Used for additional configuration of Matplotlib.
matplotlib.use('Agg')  # Configures Matplotlib to use a non-interactive backend for saving plots without displaying them.

# Define a class for breeding strategies in genetic algorithms
class BreedingStrategy:
    @staticmethod
    def two_point_crossover(parent1: SimpleModel, parent2: SimpleModel) -> SimpleModel:
        # Two-point crossover method for generating offspring from two parent models.
        child = SimpleModel()  # Create a new child model.
        
        # Hidden weights crossover
        h_size = parent1.hidden_weights.size  # Total number of elements in hidden weights.
        cut1 = random.randint(1, h_size-2)  # Randomly choose the first cut point.
        cut2 = random.randint(cut1+1, h_size-1)  # Randomly choose the second cut point after the first.

        # Flatten weights and splice based on cut points
        h_flat1 = parent1.hidden_weights.flatten()
        h_flat2 = parent2.hidden_weights.flatten()
        child_h = np.concatenate([h_flat1[:cut1], h_flat2[cut1:cut2], h_flat1[cut2:]])
        child.hidden_weights = child_h.reshape(parent1.hidden_weights.shape)  # Reshape back to original dimensions.

        # Output weights crossover
        o_size = parent1.output_weights.size
        cut1 = random.randint(1, o_size-2)
        cut2 = random.randint(cut1+1, o_size-1)
        
        o_flat1 = parent1.output_weights.flatten()
        o_flat2 = parent2.output_weights.flatten()
        child_o = np.concatenate([o_flat1[:cut1], o_flat2[cut1:cut2], o_flat1[cut2:]])
        child.output_weights = child_o.reshape(parent1.output_weights.shape)

        return child  # Return the resulting child model.

    @staticmethod
    def uniform_crossover(parent1: SimpleModel, parent2: SimpleModel) -> SimpleModel:
        # Uniform crossover randomly selects genes from either parent.
        child = SimpleModel()
        for weights_name in ['hidden_weights', 'output_weights']:
            p1_weights = getattr(parent1, weights_name)  # Get parent1's weights.
            p2_weights = getattr(parent2, weights_name)  # Get parent2's weights.
            mask = np.random.random(p1_weights.shape) < 0.5  # Create a mask to select from either parent randomly.
            child_weights = np.where(mask, p1_weights, p2_weights)  # Combine weights based on the mask.
            setattr(child, weights_name, child_weights)  # Set the child's weights.
        return child

    @staticmethod
    def blend_crossover(parent1: SimpleModel, parent2: SimpleModel, alpha: float = 0.5) -> SimpleModel:
        # Blend crossover creates a weighted combination of parent weights.
        child = SimpleModel()
        for weights_name in ['hidden_weights', 'output_weights']:
            p1_weights = getattr(parent1, weights_name)
            p2_weights = getattr(parent2, weights_name)
            blend = np.random.random(p1_weights.shape) * alpha  # Generate a random blend factor.
            child_weights = p1_weights * blend + p2_weights * (1 - blend)  # Blend weights from both parents.
            setattr(child, weights_name, child_weights)
        return child

# Define a class for fitness functions
class FitnessFunction:
    @staticmethod
    def basic_fitness(food_eaten: int, total_steps: int) -> float:
        # Calculate fitness based on food eaten and efficiency of movement.
        base_fitness = food_eaten * 75  # Assign a base fitness score for food eaten.
        efficiency_bonus = (food_eaten / total_steps) * 75 if total_steps > 0 and food_eaten > 0 else 0
        fitness = base_fitness + efficiency_bonus  # Combine base fitness with efficiency bonus.
        return fitness if food_eaten > 0 else fitness * 0.5  # Penalize if no food was eaten.

    @staticmethod
    def survival_focused_fitness(food_eaten: int, total_steps: int) -> float:
        # Focus fitness on survival duration and food eaten.
        base_fitness = food_eaten * 50
        survival_bonus = total_steps * 2  # Reward for surviving longer.
        return base_fitness + survival_bonus

    @staticmethod
    def exploration_focused_fitness(food_eaten: int, total_steps: int) -> float:
        # Focus fitness on exploration and penalize excessive steps.
        base_fitness = food_eaten * 100
        exploration_penalty = total_steps * 0.5  # Penalize for taking too many steps.
        return base_fitness - exploration_penalty

# Define a class for selection methods
class SelectionMethod:
    @staticmethod
    def tournament_selection(population: List[SimpleModel], fitness_scores: List[float], k: int = 4) -> SimpleModel:
        # Select the best individual from a random subset of the population.
        selected_indices = np.random.choice(len(population), size=k, replace=False)
        best_idx = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
        return population[best_idx]

    @staticmethod
    def roulette_wheel_selection(population: List[SimpleModel], fitness_scores: List[float]) -> SimpleModel:
        # Select individuals based on their proportional fitness.
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(population)  # Handle edge case where all fitness scores are zero.
        pick = random.uniform(0, total_fitness)
        current = 0
        for model, fitness in zip(population, fitness_scores):
            current += fitness
            if current > pick:
                return model
        return population[-1]

    @staticmethod
    def rank_selection(population: List[SimpleModel], fitness_scores: List[float]) -> SimpleModel:
        # Select individuals based on their rank in fitness scores.
        ranked_indices = np.argsort(fitness_scores)
        ranks = np.arange(1, len(population) + 1)  # Assign ranks to individuals.
        total_rank = sum(ranks)
        pick = random.uniform(0, total_rank)
        current = 0
        for rank, idx in enumerate(ranked_indices, 1):
            current += rank
            if current > pick:
                return population[idx]
        return population[ranked_indices[-1]]

# Define the ExperimentTracker class
class ExperimentTracker:
    def __init__(self, experiment_name: str):
        """
        Initialize the experiment tracker with a unique timestamped directory and store experiment details.
        
        Args:
            experiment_name (str): Name of the experiment.
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(f"experiments_{self.timestamp}")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.results = []
        self.generation_data = {}
        self.best_performances = {}

    def save_generation_data(self, combination_id: str, generation: int, 
                             avg_food: float, max_food: int, avg_fitness: float, 
                             best_fitness: float, best_food: int):
        """
        Save generation-specific data for a particular combination of strategies.
        
        Args:
            combination_id (str): Unique identifier for the strategy combination.
            generation (int): Current generation number.
            avg_food (float): Average food collected by the population.
            max_food (int): Maximum food collected in the generation.
            avg_fitness (float): Average fitness of the population.
            best_fitness (float): Best fitness achieved in the generation.
            best_food (int): Most food collected by an individual in the generation.
        """
        if combination_id not in self.generation_data:
            self.generation_data[combination_id] = []
            self.best_performances[combination_id] = {
                'best_fitness': float('-inf'),
                'best_food': 0
            }
        
        self.generation_data[combination_id].append({
            'generation': generation,
            'avg_food': avg_food,
            'max_food': max_food,
            'avg_fitness': avg_fitness,
            'best_fitness': best_fitness,
            'best_food': best_food
        })
        
        # Update best performance metrics
        if best_fitness > self.best_performances[combination_id]['best_fitness']:
            self.best_performances[combination_id]['best_fitness'] = best_fitness
        if best_food > self.best_performances[combination_id]['best_food']:
            self.best_performances[combination_id]['best_food'] = best_food

    def save_experiment_result(self, combination_id: str, config: Dict[str, Any], 
                               training_time: float):
        """
        Save the overall result for a specific experiment configuration.
        
        Args:
            combination_id (str): Unique identifier for the strategy combination.
            config (Dict[str, Any]): Configuration of the experiment.
            training_time (float): Time taken to complete the training.
        """
        best_performance = self.best_performances[combination_id]
        self.results.append({
            'combination_id': combination_id,
            'config': config,
            'best_fitness': best_performance['best_fitness'],
            'best_food': best_performance['best_food'],
            'training_time': training_time
        })

    def plot_generation_data(self):
        """
        Generate and save plots to visualize generation data and strategy comparisons.
        """
        plt.figure(figsize=(15, 10))
        
        for combination_id, data in self.generation_data.items():
            df = pd.DataFrame(data)
            plt.plot(df['generation'], df['max_food'], label=combination_id)
            
        plt.title('Maximum Food Count per Generation Across Different Combinations')
        plt.xlabel('Generation')
        plt.ylabel('Maximum Food Count')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.base_dir / 'generation_comparison.png')
        plt.close()
        
        results_df = pd.DataFrame(self.results)
        pivot_data = {}
        
        for result in self.results:
            config = result['config']
            breeding = config['breeding_strategy'].__name__
            fitness = config['fitness_function'].__name__
            selection = config['selection_method'].__name__
            key = (breeding, fitness, selection)
            pivot_data[key] = result['best_food']
            
        strategies = list(set(k[0] for k in pivot_data.keys()))
        functions = list(set(k[1] for k in pivot_data.keys()))
        methods = list(set(k[2] for k in pivot_data.keys()))
        
        for metric in ['best_food', 'best_fitness', 'training_time']:
            plt.figure(figsize=(15, 10))
            data = [[pivot_data.get((s, f, m), 0) for m in methods] for s in strategies]
            sns.heatmap(data, annot=True, fmt='.1f', 
                       xticklabels=methods, 
                       yticklabels=strategies,
                       cmap='YlOrRd')
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.tight_layout()
            plt.savefig(self.base_dir / f'{metric}_heatmap.png')
            plt.close()
        
    def save_results(self):
        """
        Save all generation data and a summary of experiment results to JSON files.
        """
        with open(self.base_dir / 'generation_data.json', 'w') as f:
            serializable_data = {}
            for key, value in self.generation_data.items():
                serializable_data[key] = value
            json.dump(serializable_data, f, indent=2)
            
        summary_results = []
        for result in self.results:
            summary_result = {
                'combination_id': result['combination_id'],
                'config': {
                    'breeding_strategy': result['config']['breeding_strategy'].__name__,
                    'fitness_function': result['config']['fitness_function'].__name__,
                    'selection_method': result['config']['selection_method'].__name__,
                    'population_size': result['config']['population_size'],
                    'mutation_rate': result['config']['mutation_rate'],
                    'mutation_scale': result['config']['mutation_scale'],
                    'elite_size': result['config']['elite_size']
                },
                'best_fitness': result['best_fitness'],
                'best_food': result['best_food'],
                'training_time': result['training_time']
            }
            summary_results.append(summary_result)
            
        with open(self.base_dir / 'summary_results.json', 'w') as f:
            json.dump(summary_results, f, indent=2)

# Define the function to evaluate fitness with a strategy
def evaluate_fitness_with_strategy(model: SimpleModel, game: SnakeGame, 
                                   fitness_function) -> Tuple[float, int]:
    """
    Evaluate the fitness of a model using the provided fitness function.
    
    Args:
        model (SimpleModel): The model to evaluate.
        game (SnakeGame): The game environment.
        fitness_function (Callable): The fitness function to calculate fitness.
        
    Returns:
        Tuple[float, int]: The fitness score and the number of food items eaten.
    """
    snake = Snake(game=game)
    food = Food(game=game)
    total_steps = 0
    food_eaten = 0
    max_steps = 200

    while total_steps < max_steps:
        state = get_input_state(snake, food, game.grid)
        action = model.get_action(state)
        snake.v = [Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0)][action]
        snake.move()

        if not snake.p.within(game.grid) or snake.cross_own_tail:
            break

        if snake.p == food.p:
            food_eaten += 1
            max_steps += 30
            snake.add_score()
            food = Food(game=game)

        total_steps += 1

    return fitness_function(food_eaten, total_steps), food_eaten

# Define the function to train a population using different strategies
def train_population_with_strategies(population_size: int,
                                   generations: int,
                                   mutation_rate: float,
                                   mutation_scale: float,
                                   elite_size: int,
                                   breeding_strategy,
                                   fitness_function,
                                   selection_method,
                                   callback=None) -> Tuple[SimpleModel, float, int]:
    """
    Train a population of models using the provided strategies and parameters.
    
    Args:
        population_size (int): Size of the population.
        generations (int): Number of generations to train.
        mutation_rate (float): Mutation rate for genetic algorithm.
        mutation_scale (float): Scale of mutations.
        elite_size (int): Number of elite individuals to retain.
        breeding_strategy (Callable): Breeding strategy function.
        fitness_function (Callable): Fitness function.
        selection_method (Callable): Selection method.
        callback (Callable, optional): Callback function to report progress.
        
    Returns:
        Tuple[SimpleModel, float, int]: The best model, its fitness score, and food count.
    """
    game = SnakeGame()
    population = [SimpleModel() for _ in range(population_size)]
    best_model = None
    best_fitness = float('-inf')
    best_food = 0

    for generation in range(generations):
        fitness_scores = []
        food_counts = []
        
        for model in population:
            fitness, food = evaluate_fitness_with_strategy(model, game, fitness_function)
            fitness_scores.append(fitness)
            food_counts.append(food)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_food = food
                best_model = SimpleModel()  # Create new model to avoid reference issues
                best_model.hidden_weights = model.hidden_weights.copy()
                best_model.output_weights = model.output_weights.copy()

        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        avg_food = sum(food_counts) / len(food_counts)
        max_gen_food = max(food_counts)

        if callback:
            callback(
                generation,
                avg_food,
                max_gen_food,
                avg_fitness,
                best_fitness,
                best_food
            )

        elite = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:elite_size]
        new_population = [model for model, _ in elite]

        while len(new_population) < population_size:
            parent1 = selection_method(population, fitness_scores)
            parent2 = selection_method(population, fitness_scores)
            child = breeding_strategy(parent1, parent2)
            child.mutate(mutation_rate, mutation_scale)
            new_population.append(child)

        population = new_population

    return best_model, best_fitness, best_food

# Define the function to run multiple experiments
def run_experiments(generations: int = 100):
    """
    Run a suite of experiments with different strategy combinations.
    
    Args:
        generations (int): Number of generations to train in each experiment.
        
    Returns:
        ExperimentTracker: Tracker object containing experiment results.
    """
    breeding_strategies = [
        BreedingStrategy.two_point_crossover,
        BreedingStrategy.uniform_crossover,
        BreedingStrategy.blend_crossover
    ]
    
    fitness_functions = [
        FitnessFunction.basic_fitness,
        FitnessFunction.survival_focused_fitness,
        FitnessFunction.exploration_focused_fitness
    ]
    
    selection_methods = [
        SelectionMethod.tournament_selection,
        SelectionMethod.roulette_wheel_selection,
        SelectionMethod.rank_selection
    ]
    
    population_size = 200
    mutation_rate = 0.05
    mutation_scale = 0.1
    elite_size = 50
    
    tracker = ExperimentTracker("snake_ai_comparison")
    log_file = tracker.base_dir / 'experiment_log.txt'
    
    def log_message(message: str):
        """Log messages to the console and a file."""
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    strategy_combinations = list(itertools.product(
        breeding_strategies,
        fitness_functions,
        selection_methods
    ))
    
    selected_combinations = strategy_combinations[:27]
    total_experiments = len(selected_combinations)
    
    initial_message = f"\nStarting Snake AI Experiment Suite\n{'='*40}\n"
    initial_message += f"\nRunning {total_experiments} different combinations..."
    initial_message += f"\nTotal generations per experiment: {generations}"
    initial_message += f"\nPopulation size: {population_size}"
    initial_message += f"\nMutation rate: {mutation_rate}"
    initial_message += f"\nMutation scale: {mutation_scale}"
    initial_message += f"\nElite size: {elite_size}\n"
    
    log_message(initial_message)
    
    for i, (breeding_strategy, fitness_function, selection_method) in enumerate(selected_combinations, 1):
        combination_id = (f"B{breeding_strategy.__name__}_"
                         f"F{fitness_function.__name__}_"
                         f"S{selection_method.__name__}")
        
        log_message(f"\nExperiment {i}/{total_experiments}")
        log_message(f"Running combination: {combination_id}")
        
        config = {
            'breeding_strategy': breeding_strategy,
            'fitness_function': fitness_function,
            'selection_method': selection_method,
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'mutation_scale': mutation_scale,
            'elite_size': elite_size
        }
        
        start_time = time.time()
        
        def training_callback(generation: int, avg_food: float, max_food: int,
                            avg_fitness: float, best_fitness: float, best_food: int):
            gen_message = f"\nGeneration {generation + 1} Stats:"
            gen_message += f"\n  Average Fitness: {avg_fitness:.2f}"
            gen_message += f"\n  Average Food: {avg_food:.2f}"
            gen_message += f"\n  Max Food This Gen: {max_food}"
            gen_message += f"\n  Best Food Ever: {best_food}"
            gen_message += f"\n  Best Fitness Overall: {best_fitness:.2f}"
            gen_message += f"\n{'-' * 40}"
            log_message(gen_message)
            
            tracker.save_generation_data(combination_id, generation, avg_food,
                                      max_food, avg_fitness, best_fitness, best_food)
        
        best_model, final_best_fitness, final_best_food = train_population_with_strategies(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            elite_size=elite_size,
            breeding_strategy=breeding_strategy,
            fitness_function=fitness_function,
            selection_method=selection_method,
            callback=training_callback
        )
        
        training_time = time.time() - start_time
        
        tracker.save_experiment_result(
            combination_id=combination_id,
            config=config,
            training_time=training_time
        )

        summary_message = f"\nCompleted {combination_id}"
        summary_message += f"\nBest fitness: {final_best_fitness:.2f}"
        summary_message += f"\nBest food count: {final_best_food}"
        summary_message += f"\nTraining time: {training_time:.2f}s"
        log_message(summary_message)
        
      tracker.plot_generation_data()
      tracker.save_results()
    
      return tracker

if __name__ == "__main__":
    GENERATIONS = 100
    tracker = run_experiments(generations=GENERATIONS)
