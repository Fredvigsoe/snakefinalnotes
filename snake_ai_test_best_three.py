# Import required modules
import time
from snake_game import SnakeGame, Snake, Food, Vector, get_input_state
from snake_ai_recording import SimpleModel
from snake_ai_testrunner import (  
    BreedingStrategy, FitnessFunction, SelectionMethod,
    train_population_with_strategies, ExperimentTracker
)

# Main function to train a model with specific strategies
def train_best_model(combination_id, breeding_strategy, fitness_function, selection_method):
    # Define training hyperparameters
    population_size = 200  # Number of models in each generation
    mutation_rate = 0.05   # Probability of mutation occurring
    mutation_scale = 0.1   # Size of mutations when they occur
    elite_size = 50        # Number of top performers to keep unchanged
    generations = 100      # Number of generations to train for

    # Initialize experiment tracking
    tracker = ExperimentTracker(f"{combination_id}_training")
    # Create log file path
    log_file = tracker.base_dir / f'{combination_id}_log.txt'

    # Helper function to log messages both to console and file
    def log_message(message: str):
        print(message)  # Print to console
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')  # Write to log file

    # Callback function that's called after each generation
    def training_callback(generation: int, avg_food: float, max_food: int,
                          avg_fitness: float, best_fitness: float, best_food: int):
        # Format generation statistics message
        gen_message = f"\nGeneration {generation + 1} Stats:"
        gen_message += f"\n  Average Fitness: {avg_fitness:.2f}"
        gen_message += f"\n  Average Food: {avg_food:.2f}"
        gen_message += f"\n  Max Food This Gen: {max_food}"
        gen_message += f"\n  Best Food Ever: {best_food}"
        gen_message += f"\n  Best Fitness Overall: {best_fitness:.2f}"
        gen_message += f"\n{'-' * 40}"
        log_message(gen_message)

        # Save generation data for later analysis
        tracker.save_generation_data(combination_id, generation, avg_food,
                                      max_food, avg_fitness, best_fitness, best_food)

    # Start training process
    start_time = time.time()
    # Train population with specified strategies
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
    # Calculate total training time
    training_time = time.time() - start_time

    # Save experiment configuration and results
    tracker.save_experiment_result(
        combination_id=combination_id,
        config={
            'breeding_strategy': breeding_strategy.__name__,
            'fitness_function': fitness_function.__name__,
            'selection_method': selection_method.__name__,
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'mutation_scale': mutation_scale,
            'elite_size': elite_size
        },
        training_time=training_time
    )

    # Generate and save plots of training progress
    tracker.plot_generation_data()
    tracker.save_results()

    # Log final summary
    summary_message = f"\nTraining completed for {combination_id}"
    summary_message += f"\nBest fitness: {final_best_fitness:.2f}"
    summary_message += f"\nBest food count: {final_best_food}"
    summary_message += f"\nTraining time: {training_time:.2f}s"
    log_message(summary_message)

# Main entry point
if __name__ == "__main__":
    # Run the best performing combination
    # This combination uses:
    # - Uniform crossover for breeding (mixing parents' genes uniformly)
    # - Basic fitness function (standard performance evaluation)
    # - Tournament selection (selecting parents through competitions)
    train_best_model(
        combination_id="Buniform_crossover_Fbasic_fitness_Stournament_selection",
        breeding_strategy=BreedingStrategy.uniform_crossover,
        fitness_function=FitnessFunction.basic_fitness,
        selection_method=SelectionMethod.tournament_selection
    )
    
    '''
    # Other combinations (commented out) for comparison:
    
    # Third best performing combination:
    # Uses exploration-focused fitness to encourage snake to explore more of the grid
    train_best_model(
        combination_id="Buniform_crossover_Fexploration_focused_fitness_Stournament_selection",
        breeding_strategy=BreedingStrategy.uniform_crossover,
        fitness_function=FitnessFunction.exploration_focused_fitness,
        selection_method=SelectionMethod.tournament_selection
    )
    
    # Second best performing combination:
    # Uses two-point crossover instead of uniform crossover
    train_best_model(
        combination_id="Btwo_point_crossover_Fbasic_fitness_Stournament_selection",
        breeding_strategy=BreedingStrategy.two_point_crossover,
        fitness_function=FitnessFunction.basic_fitness,
        selection_method=SelectionMethod.tournament_selection
    )
    '''
