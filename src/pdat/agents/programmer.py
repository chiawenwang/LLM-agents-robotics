"""
Programmer Agent: Computational modeling and numerical tools.
"""
import json
import random
from typing import Dict
from agents.base import Agent


class ProgrammerAgent(Agent):
    """
    Manages computational modeling using DDG + Neural ODE framework.
    Uses LLM to guide model architecture and training decisions.
    """

    def __init__(self, api_client=None):
        super().__init__(name="Programmer", api_client=api_client)

    def get_system_prompt(self) -> str:
        return """You are the Programmer agent in a multi-agent research system for building digital twins of physical systems.

Your role is to:
1. Develop and train neural network models of physical systems
2. Use physics-informed approaches (DDG + Neural ODE)
3. Evaluate model performance
4. Suggest improvements to model architecture or training

Technical context:
- Discrete Differential Geometry (DDG) for geometric modeling
- Neural ODEs for learning constitutive behavior
- Physics-informed neural networks (PINNs)
- Graph neural networks (GNNs) for flexible structures

You must respond with a JSON object containing:
{
    "model_architecture": {
        "type": "DDG-NeuralODE-GNN or similar",
        "hidden_dims": [layer sizes],
        "activation": "relu/tanh/...",
        "num_parameters": estimated int
    },
    "training_config": {
        "epochs": int,
        "learning_rate": float,
        "batch_size": int,
        "loss_function": "MSE/physics-informed/..."
    },
    "expected_improvements": "what aspects should improve",
    "validation_strategy": "how to evaluate the model",
    "rationale": "why this approach is appropriate"
}

Focus on creating models that:
- Respect known physics (conservation laws, geometry)
- Learn efficiently from limited data
- Generalize well to unseen configurations"""

    def step(self, input_data: Dict) -> Dict:
        """
        Update/train the digital twin model.
        
        Args:
            input_data: Context including iteration, experiment data
            
        Returns:
            Model results dictionary with validation metrics
        """
        iteration = input_data.get("iteration", 0)
        experiment_results = input_data.get("experiment", {})
        
        self.log(f"Developing model for iteration {iteration}")
        
        # Build context for model development
        model_prompt = f"""
MODEL DEVELOPMENT REQUEST:

Current iteration: {iteration}
Available training data: {input_data.get('num_samples', 'Unknown')} samples
Data path: {input_data.get('data_path', 'Unknown')}

Task: Design or update a neural network model for the Slinky digital twin.

"""
        
        # Add guidance based on iteration
        if iteration == 0:
            model_prompt += """
This is the FIRST model. Design:
- Initial architecture that balances capacity and efficiency
- Training strategy for the available data
- Validation approach to measure progress
"""
        else:
            model_prompt += f"""
This is iteration {iteration}. Consider:
- How to incorporate new data effectively
- Whether to expand model capacity or adjust training
- What validation metrics matter most now
"""
        
        # Use LLM for model design (if available)
        if self.api_client:
            try:
                response = self.call_llm(
                    user_message=model_prompt,
                    include_memory=True,
                    max_tokens=1000,
                    temperature=0.4  # Lower temperature for technical decisions
                )
                
                model_config = self._parse_model_config(response)
                
            except Exception as e:
                self.log(f"LLM model design failed, using default: {e}", "WARNING")
                model_config = self._default_model_config(iteration)
        else:
            model_config = self._default_model_config(iteration)
        
        # Train the model (mock for now)
        self.log(f"Training model: {model_config['model_architecture']['type']}")
        
        results = self._train_model(model_config, input_data)
        
        # Add to memory
        self.add_to_memory({
            "type": "model_update",
            "content": results
        })
        
        return results

    def _parse_model_config(self, response: str) -> Dict:
        """
        Parse LLM response into model configuration.
        """
        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                config = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["model_architecture", "training_config"]
                for field in required_fields:
                    if field not in config:
                        raise ValueError(f"Missing required field: {field}")
                
                return config
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            self.log(f"Failed to parse model config: {e}", "ERROR")
            self.log(f"Raw response: {response}", "DEBUG")
            raise

    def _default_model_config(self, iteration: int) -> Dict:
        """
        Fallback model configuration.
        """
        # Gradually increase model capacity
        base_dim = 64
        hidden_dims = [base_dim * (2 ** i) for i in range(3)]  # [64, 128, 256]
        
        return {
            "model_architecture": {
                "type": "DDG-NeuralODE-GNN",
                "hidden_dims": hidden_dims,
                "activation": "relu",
                "num_parameters": sum(hidden_dims) * 10  # Rough estimate
            },
            "training_config": {
                "epochs": 100 + iteration * 50,  # More training as we go
                "learning_rate": 0.001 / (1 + iteration * 0.1),  # Decay over time
                "batch_size": 32,
                "loss_function": "physics-informed-MSE"
            },
            "expected_improvements": f"Iteration {iteration} improvements",
            "validation_strategy": "k-fold cross-validation",
            "rationale": f"Standard configuration for iteration {iteration}"
        }

    def _train_model(self, config: Dict, input_data: Dict) -> Dict:
        """
        Train the model (mock implementation).
        
        In real implementation, this would:
        1. Load data from experiment results
        2. Build neural network with specified architecture
        3. Train using physics-informed loss
        4. Validate on held-out data
        5. Save model checkpoint
        """
        self.log("Simulating model training...", "DEBUG")
        
        # Mock training
        import time
        
        epochs = config["training_config"].get("epochs", 100)
        
        # Simulate some processing time
        time.sleep(0.5)
        
        # Simulate improving validation error over iterations
        iteration = input_data.get("iteration", 0)
        
        # Start with high error, decrease with each iteration
        # Add some randomness to simulate realistic training
        base_error = 0.5 / (1 + iteration)
        validation_error = base_error * random.uniform(0.8, 1.2)
        
        # Add some noise but ensure general downward trend
        training_error = validation_error * random.uniform(0.6, 0.9)
        
        results = {
            "model_config": config,
            "training_error": training_error,
            "validation_error": validation_error,
            "epochs_trained": epochs,
            "training_time": 0.5,
            "model_path": f"/models/slinky_iter_{iteration}.pt",
            "metrics": {
                "mae": validation_error * 0.8,
                "rmse": validation_error,
                "r2_score": 1 - validation_error
            }
        }
        
        self.log(
            f"Training complete: Val Error = {validation_error:.4f}, "
            f"Train Error = {training_error:.4f}"
        )
        
        return results