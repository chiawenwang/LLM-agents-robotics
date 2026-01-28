"""
Demo script showing how to use individual agents and test the framework.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from api_client import ClaudeAPIClient
from agents.supervisor import SupervisorAgent
from agents.scientist import ScientistAgent
from agents.programmer import ProgrammerAgent


def demo_mock_mode():
    """
    Demonstrate the framework in mock mode (no API calls).
    Good for testing the structure without using credits.
    """
    print("\n" + "="*80)
    print("  DEMO: Mock Mode (No API Calls)")
    print("="*80 + "\n")
    
    # Initialize agents without API client
    supervisor = SupervisorAgent(api_client=None, error_threshold=0.1, max_iterations=3)
    scientist = ScientistAgent(api_client=None)
    programmer = ProgrammerAgent(api_client=None)
    
    # Test state
    state = {
        "objective": "Test the framework",
        "iteration": 0,
        "model_error": None,
        "done": False
    }
    
    print("Testing Supervisor decision making...")
    decision = supervisor.step(state)
    print(f"✓ Decision: {decision['action']}")
    print(f"  Reason: {decision['reason']}\n")
    
    print("Testing Scientist experiment design...")
    experiment = scientist.step(state)
    print(f"✓ Experiment type: {experiment['experiment']['experiment_type']}")
    print(f"  Samples: {experiment['num_samples']}\n")
    
    print("Testing Programmer model development...")
    model = programmer.step({"iteration": 0, **experiment})
    print(f"✓ Model architecture: {model['model_config']['model_architecture']['type']}")
    print(f"  Validation error: {model['validation_error']:.4f}\n")
    
    print("Mock mode test complete! ✓\n")


def demo_with_api():
    """
    Demonstrate the framework with Claude API.
    Runs a minimal test with 2 iterations to keep costs low.
    """
    print("\n" + "="*80)
    print("  DEMO: With Claude API")
    print("="*80 + "\n")
    
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set. Run demo_mock_mode() instead.\n")
        return
    
    # Initialize API client with Haiku (cheapest)
    try:
        api_client = ClaudeAPIClient(
            api_key=api_key,
            model="claude-haiku-4-5-20251001"
        )
        print("✓ Claude API client initialized (Haiku model)")
    except Exception as e:
        print(f"❌ Failed to initialize API: {e}\n")
        return
    
    # Initialize agents with API
    supervisor = SupervisorAgent(
        api_client=api_client,
        error_threshold=0.1,
        max_iterations=2  # Keep it short for demo
    )
    scientist = ScientistAgent(api_client=api_client)
    programmer = ProgrammerAgent(api_client=api_client)
    
    print("✓ All agents initialized\n")
    
    # Run one iteration
    state = {
        "objective": "Test Claude API integration",
        "iteration": 0,
        "model_error": None,
        "done": False
    }
    
    print("-" * 80)
    print("Running iteration 0...")
    print("-" * 80 + "\n")
    
    # Supervisor decision
    print("1. Supervisor making decision...")
    decision = supervisor.step(state)
    print(f"   ✓ Action: {decision['action']}")
    print(f"   ✓ Reason: {decision['reason']}")
    
    if decision["action"] == "stop":
        print("\nDemo complete (supervisor stopped).\n")
        return
    
    # Scientist experiment
    print("\n2. Scientist designing experiment...")
    experiment = scientist.step(state)
    print(f"   ✓ Type: {experiment['experiment']['experiment_type']}")
    
    # Programmer model
    print("\n3. Programmer developing model...")
    model = programmer.step({"iteration": 0, **experiment})
    print(f"   ✓ Validation error: {model['validation_error']:.4f}")
    
    # Show cost
    print("\n" + "-" * 80)
    cost_info = api_client.get_cost_estimate()
    print(f"💰 Cost for this demo:")
    print(f"   Tokens used: {cost_info['input_tokens'] + cost_info['output_tokens']:,}")
    print(f"   Estimated cost: ${cost_info['total_cost_usd']:.4f} USD")
    print("-" * 80 + "\n")
    
    print("API demo complete! ✓\n")


def demo_agent_memory():
    """
    Demonstrate how agent memory works.
    """
    print("\n" + "="*80)
    print("  DEMO: Agent Memory System")
    print("="*80 + "\n")
    
    agent = SupervisorAgent(api_client=None)
    
    print("Adding memories to agent...")
    
    # Add some sample memories
    agent.add_to_memory({
        "type": "decision",
        "decision": {"action": "run_experiment", "reason": "First iteration"}
    })
    
    agent.add_to_memory({
        "type": "experiment_result",
        "content": {"status": "completed", "data_path": "/data/run1"}
    })
    
    agent.add_to_memory({
        "type": "model_update",
        "content": {"validation_error": 0.234}
    })
    
    print(f"\n✓ Agent has {len(agent.memory)} memories\n")
    
    # Show memory context
    print("Memory context (formatted for LLM):")
    print("-" * 80)
    context = agent.get_memory_context(max_entries=3)
    print(context)
    print("-" * 80 + "\n")
    
    print("Memory demo complete! ✓\n")


def main():
    """
    Run all demos.
    """
    print("\n" + "="*80)
    print("  MULTI-AGENT FRAMEWORK DEMOS")
    print("="*80)
    
    # Always run mock mode demo
    demo_mock_mode()
    
    # Show memory demo
    demo_agent_memory()
    
    # Check if API key is available
    if os.environ.get("ANTHROPIC_API_KEY"):
        response = input("Run demo with Claude API? (costs ~$0.001) [y/N]: ")
        if response.lower() == 'y':
            demo_with_api()
        else:
            print("\nSkipping API demo. Set ANTHROPIC_API_KEY and run again to test.\n")
    else:
        print("="*80)
        print("  To test with Claude API:")
        print("  1. Set ANTHROPIC_API_KEY environment variable")
        print("  2. Run this demo again")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
