"""
Supervisor Agent: High-level planning and orchestration.
"""
import json
from typing import Dict
from agents.base import Agent


class SupervisorAgent(Agent):
    """
    High-level planning and task orchestration.
    Uses LLM to make intelligent decisions about project progress.
    """

    def __init__(self, api_client=None, error_threshold: float = 0.05, max_iterations: int = 10):
        super().__init__(name="Supervisor", api_client=api_client)
        self.error_threshold = error_threshold
        self.max_iterations = max_iterations

    def get_system_prompt(self) -> str:
        return f"""You are the Supervisor agent in a multi-agent research system for building digital twins of physical systems.

Your role is to:
1. Evaluate the current state of the project
2. Decide what action to take next
3. Coordinate between the Scientist (experiments) and Programmer (modeling) agents
4. Determine when the project objective has been achieved

Project parameters:
- Error threshold: {self.error_threshold}
- Max iterations: {self.max_iterations}

You must respond with a JSON object containing:
{{
    "action": "run_experiment" or "stop",
    "reason": "brief explanation of your decision",
    "priority": "what the team should focus on next (if continuing)",
    "observations": "key insights from current state"
}}

Decision criteria:
- If no model exists yet → run_experiment
- If model error > threshold → run_experiment (gather more data)
- If model error ≤ threshold → stop (objective achieved)
- If max iterations reached → stop (budget exhausted)

Be concise but insightful in your reasoning."""

    def step(self, state: Dict) -> Dict:
        """
        Evaluate current state and decide next action.
        
        Args:
            state: Current project state including iteration, model_error, objective
            
        Returns:
            Decision dictionary with action, reason, and metadata
        """
        self.log(f"Evaluating state at iteration {state['iteration']}/{self.max_iterations}")
        
        # Build context for LLM
        state_summary = f"""
PROJECT STATE:
- Objective: {state.get('objective', 'Unknown')}
- Current Iteration: {state['iteration']}/{self.max_iterations}
- Model Error: {state.get('model_error', 'None (no model yet)')}
- Error Threshold: {self.error_threshold}

Please analyze this state and decide the next action.
"""
        
        # Use LLM for decision making (if available)
        if self.api_client:
            try:
                response = self.call_llm(
                    user_message=state_summary,
                    include_memory=True,
                    max_tokens=500,
                    temperature=0.3  # Lower temperature for more deterministic decisions
                )
                
                # Parse JSON response
                decision = self._parse_llm_decision(response)
                
            except Exception as e:
                self.log(f"LLM decision failed, falling back to heuristics: {e}", "WARNING")
                decision = self._heuristic_decision(state)
        else:
            # Fallback to rule-based logic
            decision = self._heuristic_decision(state)
        
        # Add to memory
        self.add_to_memory({
            "type": "decision",
            "state_snapshot": dict(state),
            "decision": decision
        })
        
        self.log(f"Decision: {decision['action']} - {decision['reason']}")
        
        return decision

    def _parse_llm_decision(self, response: str) -> Dict:
        """
        Parse LLM response into decision dictionary.
        """
        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                decision = json.loads(json_str)
                
                # Validate required fields
                if "action" not in decision or "reason" not in decision:
                    raise ValueError("Missing required fields in LLM response")
                
                # Validate action value
                if decision["action"] not in ["run_experiment", "stop"]:
                    raise ValueError(f"Invalid action: {decision['action']}")
                
                return decision
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            self.log(f"Failed to parse LLM response: {e}", "ERROR")
            self.log(f"Raw response: {response}", "DEBUG")
            raise

    def _heuristic_decision(self, state: Dict) -> Dict:
        """
        Fallback rule-based decision making.
        """
        # Stop if max iterations reached
        if state["iteration"] >= self.max_iterations:
            return {
                "action": "stop",
                "reason": "Max iterations reached",
                "priority": "None",
                "observations": "Budget exhausted"
            }

        model_error = state.get("model_error")

        # First iteration: no model yet
        if model_error is None:
            return {
                "action": "run_experiment",
                "reason": "No model exists yet - need initial data",
                "priority": "Collect diverse training data",
                "observations": "Starting fresh"
            }

        # Model not good enough → more data
        elif model_error > self.error_threshold:
            return {
                "action": "run_experiment",
                "reason": f"Model error {model_error:.4f} above threshold {self.error_threshold}",
                "priority": "Focus on regions with high error",
                "observations": f"Need {((model_error/self.error_threshold - 1) * 100):.1f}% improvement"
            }

        # Model is good enough → stop
        else:
            return {
                "action": "stop",
                "reason": f"Model error {model_error:.4f} below threshold {self.error_threshold}",
                "priority": "Validation and deployment",
                "observations": "Objective achieved"
            }