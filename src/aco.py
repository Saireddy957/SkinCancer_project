"""
Ant Colony Optimization (ACO) for Hyperparameter Tuning
Bio-inspired optimization algorithm for finding optimal model hyperparameters
"""

import numpy as np
import torch
from copy import deepcopy
from train import train
from evaluate import evaluate_model


class AntColonyOptimizer:
    """
    Ant Colony Optimization for hyperparameter tuning
    """
    
    def __init__(self, param_space, n_ants=10, n_iterations=20, 
                 evaporation_rate=0.1, alpha=1.0, beta=2.0):
        """
        Args:
            param_space (dict): Dictionary defining the hyperparameter search space
                Example: {
                    'learning_rate': [0.001, 0.0001, 0.00001],
                    'batch_size': [16, 32, 64],
                    'backbone': ['resnet50', 'efficientnet_b0']
                }
            n_ants (int): Number of ants (solutions) per iteration
            n_iterations (int): Number of iterations
            evaporation_rate (float): Pheromone evaporation rate
            alpha (float): Importance of pheromone trail
            beta (float): Importance of heuristic information
        """
        self.param_space = param_space
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        
        # Initialize pheromone matrix
        self.pheromones = {}
        for param, values in param_space.items():
            self.pheromones[param] = np.ones(len(values))
        
        # Track best solution
        self.best_params = None
        self.best_score = 0.0
        self.history = []
    
    def _select_param_value(self, param_name):
        """
        Select a parameter value based on pheromone levels
        """
        values = self.param_space[param_name]
        pheromones = self.pheromones[param_name]
        
        # Calculate probabilities using pheromone levels
        probabilities = pheromones ** self.alpha
        probabilities /= probabilities.sum()
        
        # Select value based on probabilities
        idx = np.random.choice(len(values), p=probabilities)
        return values[idx], idx
    
    def _construct_solution(self):
        """
        Construct a solution (parameter configuration) for an ant
        """
        solution = {}
        indices = {}
        
        for param_name in self.param_space.keys():
            value, idx = self._select_param_value(param_name)
            solution[param_name] = value
            indices[param_name] = idx
        
        return solution, indices
    
    def _evaluate_solution(self, params, base_config):
        """
        Evaluate a solution by training a model with given parameters
        
        Returns:
            float: Validation accuracy score
        """
        # Update config with ACO parameters
        config = deepcopy(base_config)
        config.update(params)
        
        try:
            # Train model with these parameters
            print(f"\nTesting parameters: {params}")
            model, history = train(config)
            
            # Return best validation accuracy
            best_val_acc = max(history['val_acc'])
            print(f"Validation accuracy: {best_val_acc:.4f}")
            
            return best_val_acc
        except Exception as e:
            print(f"Error during training: {e}")
            return 0.0
    
    def _update_pheromones(self, solutions, scores):
        """
        Update pheromone levels based on solution quality
        """
        # Evaporate pheromones
        for param in self.pheromones:
            self.pheromones[param] *= (1 - self.evaporation_rate)
        
        # Deposit new pheromones
        for solution, score in zip(solutions, scores):
            for param, idx in solution['indices'].items():
                # Deposit pheromone proportional to solution quality
                self.pheromones[param][idx] += score
    
    def optimize(self, base_config):
        """
        Run ACO optimization
        
        Args:
            base_config (dict): Base training configuration
        
        Returns:
            dict: Best hyperparameters found
            float: Best score achieved
        """
        print("="*60)
        print("Starting Ant Colony Optimization for Hyperparameter Tuning")
        print("="*60)
        print(f"Parameter space: {self.param_space}")
        print(f"Number of ants: {self.n_ants}")
        print(f"Number of iterations: {self.n_iterations}")
        print("="*60)
        
        for iteration in range(self.n_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.n_iterations}")
            print(f"{'='*60}")
            
            solutions = []
            scores = []
            
            # Generate and evaluate solutions for all ants
            for ant in range(self.n_ants):
                print(f"\n--- Ant {ant + 1}/{self.n_ants} ---")
                
                # Construct solution
                params, indices = self._construct_solution()
                
                # Evaluate solution
                score = self._evaluate_solution(params, base_config)
                
                solutions.append({'params': params, 'indices': indices})
                scores.append(score)
                
                # Update best solution
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    print(f"✓ New best solution found! Score: {score:.4f}")
            
            # Update pheromones
            self._update_pheromones(solutions, scores)
            
            # Track history
            self.history.append({
                'iteration': iteration + 1,
                'best_score': self.best_score,
                'avg_score': np.mean(scores),
                'best_params': deepcopy(self.best_params)
            })
            
            print(f"\nIteration {iteration + 1} Summary:")
            print(f"  Average Score: {np.mean(scores):.4f}")
            print(f"  Best Score So Far: {self.best_score:.4f}")
            print(f"  Best Params: {self.best_params}")
        
        print("\n" + "="*60)
        print("ACO Optimization Completed!")
        print("="*60)
        print(f"Best Score: {self.best_score:.4f}")
        print(f"Best Parameters: {self.best_params}")
        print("="*60)
        
        return self.best_params, self.best_score


def run_aco_optimization():
    """
    Example function to run ACO optimization
    """
    # Define parameter search space
    param_space = {
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [16, 32, 64],
        'backbone': ['resnet50', 'efficientnet_b0', 'mobilenet_v2']
    }
    
    # Base configuration
    base_config = {
        'data_dir': '/content/images',
        'metadata_file': '../data/metadata.csv',
        'save_dir': '../models/aco',
        'model_type': 'multimodal',
        'num_classes': 7,
        'pretrained': True,
        'num_epochs': 10,  # Reduce epochs for faster ACO search
        'num_workers': 4
    }
    
    # Create ACO optimizer
    aco = AntColonyOptimizer(
        param_space=param_space,
        n_ants=5,
        n_iterations=5,
        evaporation_rate=0.1,
        alpha=1.0,
        beta=2.0
    )
    
    # Run optimization
    best_params, best_score = aco.optimize(base_config)
    
    return best_params, best_score, aco.history


if __name__ == "__main__":
    best_params, best_score, history = run_aco_optimization()
    
    print("\nFinal Results:")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Validation Score: {best_score:.4f}")
