"""
Main application for multi-agent digital twin research framework.
"""
import os
import sys
from typing import Optional

from pathlib import Path

# Add parent directory to path so we can import from pdat/
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from api_client import ClaudeAPIClient
from agents.supervisor import SupervisorAgent
from agents.scientist import ScientistAgent
from agents.programmer import ProgrammerAgent

def print_separator(char='=', length=80):
    """Print a visual separator."""
    print(char * length)


def print_header(text: str):
    """Print a formatted header."""
    print_separator()
    print(f"  {text}")
    print_separator()


def main(use_llm: bool = None, model: str = None):
    """
    Run the multi-agent digital twin research framework.
    
    Args:
        use_llm: Whether to use Claude API (defaults to config.USE_LLM)
        model: Which Claude model to use (defaults to config.CLAUDE_MODEL)
    """
    # Use config defaults if not specified
    if use_llm is None:
        use_llm = config.USE_LLM
    if model is None:
        model = config.CLAUDE_MODEL
    
    print_header("MULTI-AGENT DIGITAL TWIN RESEARCH FRAMEWORK")
    
    # Initialize API client if requested
    api_client = None
    if use_llm:
        try:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("\n⚠️  WARNING: ANTHROPIC_API_KEY not found in environment")
                print("   Running in mock mode without LLM integration\n")
                use_llm = False
            else:
                api_client = ClaudeAPIClient(api_key=api_key, model=model)
                print(f"\n✓ Claude API initialized (model: {model})")
                print(f"  Cost tracking enabled\n")
        except Exception as e:
            print(f"\n⚠️  WARNING: Failed to initialize Claude API: {e}")
            print("   Running in mock mode without LLM integration\n")
            use_llm = False
    else:
        print("\n⚠️  Running in MOCK MODE (no LLM calls)")
        print("   Set use_llm=True and ANTHROPIC_API_KEY to enable Claude integration\n")
    
    # Initialize agents
    print("Initializing agents...")
    supervisor = SupervisorAgent(
        api_client=api_client,
        error_threshold=config.ERROR_THRESHOLD,
        max_iterations=config.MAX_ITERATIONS
    )
    scientist = ScientistAgent(api_client=api_client)
    programmer = ProgrammerAgent(api_client=api_client)
    print("✓ All agents initialized\n")
    
    # Initialize project state
    state = {
        "objective": config.PROJECT_OBJECTIVE,
        "iteration": 0,
        "model_error": None,
        "done": False
    }
    
    print_header("STARTING PROJECT")
    print(f"Objective: {state['objective']}")
    print(f"Max iterations: {supervisor.max_iterations}")
    print(f"Error threshold: {supervisor.error_threshold}\n")
    
    # Main agent loop
    try:
        while not state["done"]:
            print_separator('-')
            print(f"\n🔄 ITERATION {state['iteration']}")
            print_separator('-')
            
            # Step 1: Supervisor makes decision
            print("\n1️⃣  SUPERVISOR: Making decision...")
            decision = supervisor.step(state)
            
            if decision["action"] == "stop":
                state["done"] = True
                print("\n✓ Supervisor decided to STOP")
                print(f"   Reason: {decision['reason']}")
                break
            
            print(f"\n✓ Supervisor decided to: {decision['action']}")
            print(f"   Reason: {decision['reason']}")
            if "priority" in decision:
                print(f"   Priority: {decision['priority']}")
            
            # Step 2: Scientist designs and runs experiment
            print("\n2️⃣  SCIENTIST: Designing experiment...")
            experiment_results = scientist.step(state)
            
            print(f"\n✓ Experiment completed")
            print(f"   Type: {experiment_results['experiment'].get('experiment_type', 'unknown')}")
            print(f"   Samples: {experiment_results.get('num_samples', 'unknown')}")
            print(f"   Data: {experiment_results.get('data_path', 'unknown')}")
            
            # Step 3: Programmer updates model
            print("\n3️⃣  PROGRAMMER: Developing model...")
            model_results = programmer.step({
                "iteration": state["iteration"],
                **experiment_results
            })
            
            print(f"\n✓ Model training completed")
            print(f"   Architecture: {model_results['model_config']['model_architecture'].get('type', 'unknown')}")
            print(f"   Validation Error: {model_results['validation_error']:.4f}")
            print(f"   Training Error: {model_results['training_error']:.4f}")
            
            # Update global state
            state["model_error"] = model_results["validation_error"]
            state["iteration"] += 1
            
            # Print current progress
            print(f"\n📊 Progress: Error {state['model_error']:.4f} vs Threshold {supervisor.error_threshold}")
            
        print_separator('-')
        print("\n" + "="*80)
        print("  PROJECT COMPLETE")
        print("="*80)
        
        # Print final summary
        print("\n📋 FINAL SUMMARY:")
        print(f"   Total iterations: {state['iteration']}")
        print(f"   Final model error: {state.get('model_error', 'N/A')}")
        print(f"   Objective: {state['objective']}")
        print(f"   Status: {'✓ Achieved' if state.get('model_error', 1) <= supervisor.error_threshold else '⚠ Incomplete'}")
        
        # Print cost summary if using LLM
        if api_client:
            print("\n💰 COST SUMMARY:")
            cost_info = api_client.get_cost_estimate()
            print(f"   Input tokens:  {cost_info['input_tokens']:,}")
            print(f"   Output tokens: {cost_info['output_tokens']:,}")
            print(f"   Total tokens:  {cost_info['input_tokens'] + cost_info['output_tokens']:,}")
            print(f"   Estimated cost: ${cost_info['total_cost_usd']:.4f} USD")
        
        # Print memory summary
        print("\n🧠 MEMORY SUMMARY:")
        print(f"   Supervisor memories: {len(supervisor.memory)}")
        print(f"   Scientist memories: {len(scientist.memory)}")
        print(f"   Programmer memories: {len(programmer.memory)}")
        
        print("\n" + "="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print_separator()
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print_separator()
        return 1
    
    return 0


if __name__ == "__main__":
    # Print configuration summary
    print(config.get_config_summary())
    
    # Check for API key if needed
    if config.USE_LLM and not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n" + "="*80)
        print("  SETUP REQUIRED")
        print("="*80)
        print("\nTo use Claude API, set your API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nOr edit config.py and set USE_LLM = False to run in mock mode\n")
        print("="*80 + "\n")
        sys.exit(1)
    
    # Run main application
    exit_code = main()
    sys.exit(exit_code)