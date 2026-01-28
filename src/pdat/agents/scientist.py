"""
Scientist Agent: Experiment design and execution.
"""
import json
from typing import Dict
from agents.base import Agent


class ScientistAgent(Agent):
    """
    Designs experiments and interacts with physical/simulated robots.
    Uses LLM to create adaptive experiment plans.
    """

    def __init__(self, api_client=None):
        super().__init__(name="Scientist", api_client=api_client)

    def get_system_prompt(self) -> str:
        return """You are the Scientist agent in a multi-agent research system for building digital twins of physical systems.

Your role is to:
1. Design experiments to collect data about physical systems (e.g., Slinky)
2. Determine what parameters to vary and measure
3. Plan data collection strategies
4. Adapt experiments based on previous results

You are currently working with a Slinky - a helical spring with complex nonlinear behavior.

Key considerations for Slinky experiments:
- Geometric parameters: stretch, compression, curvature, twist
- Material properties: stiffness, damping, friction
- Dynamic behavior: oscillations, wave propagation, collapse
- Data to collect: node positions, forces, video, timestamps

You must respond with a JSON object containing:
{
    "experiment_type": "descriptive name",
    "parameters": {
        "stretch_range": [min, max],
        "curvature_range": [min, max],
        "num_trials": int,
        "other_params": "..."
    },
    "data_to_collect": ["video", "positions", "forces", ...],
    "rationale": "why this experiment will help improve the model",
    "expected_insights": "what we expect to learn"
}

Design experiments that will efficiently explore the parameter space and improve model accuracy."""

    def step(self, input_data: Dict) -> Dict:
        """
        Design and execute an experiment.
        
        Args:
            input_data: Context including iteration, previous results, model error
            
        Returns:
            Experiment results dictionary
        """
        iteration = input_data.get("iteration", 0)
        model_error = input_data.get("model_error")
        
        self.log(f"Designing experiment for iteration {iteration}")
        
        # Build context for experiment design
        experiment_prompt = f"""
EXPERIMENT DESIGN REQUEST:

Current iteration: {iteration}
Model error (if exists): {model_error}

Task: Design an experiment to collect data about Slinky behavior.

"""
        
        # Add guidance based on iteration
        if iteration == 0:
            experiment_prompt += """
This is the FIRST experiment. Design a broad exploration to:
- Cover a wide range of Slinky configurations
- Establish baseline behavior
- Test multiple deformation modes (stretch, bend, twist)
"""
        else:
            experiment_prompt += f"""
This is iteration {iteration}. Consider:
- What worked/didn't work in previous experiments
- Where is the model currently struggling?
- What parameter regions need more data?
"""
        
        # Use LLM for experiment design (if available)
        if self.api_client:
            try:
                response = self.call_llm(
                    user_message=experiment_prompt,
                    include_memory=True,
                    max_tokens=800,
                    temperature=0.5  # Moderate temperature for creativity
                )
                
                experiment = self._parse_experiment_design(response)
                
            except Exception as e:
                self.log(f"LLM experiment design failed, using default: {e}", "WARNING")
                experiment = self._default_experiment(iteration)
        else:
            experiment = self._default_experiment(iteration)
        
        # Execute the experiment (mock for now)
        self.log(f"Executing experiment: {experiment['experiment_type']}")
        
        results = self._execute_experiment(experiment)
        
        # Add to memory
        self.add_to_memory({
            "type": "experiment_result",
            "content": results
        })
        
        return results

    def _parse_experiment_design(self, response: str) -> Dict:
        """
        Parse LLM response into experiment specification.
        """
        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                experiment = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["experiment_type", "parameters", "data_to_collect"]
                for field in required_fields:
                    if field not in experiment:
                        raise ValueError(f"Missing required field: {field}")
                
                return experiment
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            self.log(f"Failed to parse experiment design: {e}", "ERROR")
            self.log(f"Raw response: {response}", "DEBUG")
            raise

    def _default_experiment(self, iteration: int) -> Dict:
        """
        Fallback experiment design.
        """
        # Vary parameters based on iteration
        num_trials = 10 + iteration * 5  # More trials as we progress
        
        return {
            "experiment_type": f"slinky_manipulation_iter_{iteration}",
            "parameters": {
                "stretch_range": [0.5 + iteration * 0.1, 1.5 - iteration * 0.05],
                "curvature_range": [-0.8, 0.8],
                "num_trials": num_trials,
                "sample_rate": 30  # Hz
            },
            "data_to_collect": [
                "video",
                "node_positions",
                "forces",
                "timestamps"
            ],
            "rationale": f"Iteration {iteration} standard experiment",
            "expected_insights": "Improve model across parameter space"
        }

    def _execute_experiment(self, experiment: Dict) -> Dict:
        """
        Execute the experiment (mock implementation).
        
        In real implementation, this would:
        1. Interface with Sawyer robot via slinky_tools
        2. Control the Slinky manipulation
        3. Collect sensor data
        4. Save to disk
        """
        self.log("Executing experiment...", "DEBUG")
        
        # PLACEHOLDER: Integration with real robot
        # Uncomment and modify when you have access to the robot:
        #
        # from slinky_tools import create_slinky_experiment
        # slinky_exp = create_slinky_experiment(mock=False)  # Set False for real robot
        # results = slinky_exp.run_parameterized_experiment(experiment)
        # return results
        
        # MOCK IMPLEMENTATION (for testing without robot):
        import time
        import random
        
        num_trials = experiment["parameters"].get("num_trials", 10)
        
        # Simulate some processing time
        time.sleep(0.5)
        
        # Generate mock data path
        data_path = f"/data/slinky/iter_{len(self.memory)}/run_{random.randint(1000, 9999)}/"
        
        results = {
            "experiment": experiment,
            "status": "completed",
            "data_path": data_path,
            "num_samples": num_trials * 100,  # Mock: 100 samples per trial
            "execution_time": 0.5,
            "quality_metrics": {
                "coverage": random.uniform(0.7, 0.95),
                "noise_level": random.uniform(0.01, 0.05)
            }
        }
        
        self.log(f"Experiment complete: {num_trials} trials, {results['num_samples']} samples")
        
        return results
