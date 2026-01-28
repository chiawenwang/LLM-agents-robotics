"""
Configuration file for the Multi-Agent Digital Twin Framework.

Edit these settings to customize the framework behavior.
"""
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Enable/disable Claude API integration
# Set to False to run in mock mode (no API calls, no costs)
USE_LLM = True

# Claude model to use
# Options:
#   "claude-haiku-4-5-20251001"   - Fastest, cheapest (~$0.002/run)
#   "claude-sonnet-4-5-20250929"  - Better quality (~$0.015/run)
#   "claude-opus-4-5-20251101"    - Best quality (~$0.10/run)
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# Project objective
PROJECT_OBJECTIVE = "Build a digital twin of a Slinky"

# Maximum number of experiment-model iterations
MAX_ITERATIONS = 10

# Model error threshold for stopping
# Stop when validation error drops below this value
ERROR_THRESHOLD = 0.05

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# LLM parameters for each agent type
SUPERVISOR_CONFIG = {
    "max_tokens": 500,
    "temperature": 0.3,  # Lower = more deterministic
}

SCIENTIST_CONFIG = {
    "max_tokens": 800,
    "temperature": 0.5,  # Moderate for creative experiment design
}

PROGRAMMER_CONFIG = {
    "max_tokens": 1000,
    "temperature": 0.4,  # Lower for technical precision
}

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Slinky experiment parameters
SLINKY_PARAMS = {
    # Range for stretching [min_ratio, max_ratio]
    "stretch_range": [0.5, 2.0],
    
    # Range for curvature [min, max] in radians
    "curvature_range": [-0.8, 0.8],
    
    # Initial number of trials per experiment
    "initial_trials": 10,
    
    # Increase trials per iteration
    "trials_increment": 5,
    
    # Sampling rate for data collection (Hz)
    "sample_rate": 30,
}

# Data collection settings
DATA_COLLECTION = {
    "collect_video": True,
    "collect_positions": True,
    "collect_forces": True,
    "collect_timestamps": True,
    "video_resolution": (640, 480),
    "video_fps": 30,
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Neural network architecture
MODEL_ARCHITECTURE = {
    # Base dimensions for hidden layers
    "base_hidden_dim": 64,
    
    # Number of hidden layers
    "num_hidden_layers": 3,
    
    # Activation function
    "activation": "relu",  # Options: relu, tanh, gelu, silu
    
    # Model type
    "type": "DDG-NeuralODE-GNN",
}

# Training configuration
TRAINING_CONFIG = {
    # Initial number of epochs
    "initial_epochs": 100,
    
    # Additional epochs per iteration
    "epochs_increment": 50,
    
    # Initial learning rate
    "learning_rate": 0.001,
    
    # Learning rate decay per iteration
    "lr_decay": 0.1,
    
    # Batch size
    "batch_size": 32,
    
    # Loss function
    "loss_function": "physics-informed-MSE",
}

# =============================================================================
# LOGGING AND OUTPUT
# =============================================================================

# Logging configuration
LOGGING = {
    # Enable detailed logging
    "verbose": True,
    
    # Log to file
    "log_to_file": False,
    "log_file": "agent_logs.txt",
    
    # Print agent reasoning
    "print_llm_responses": False,  # Set True to see full LLM outputs
}

# Output directories
OUTPUT_PATHS = {
    "data_dir": "/data/slinky",
    "model_dir": "/models",
    "log_dir": "/logs",
}

# =============================================================================
# ISAAC SIM INTEGRATION (Future)
# =============================================================================

# NVIDIA Isaac Sim settings
ISAAC_SIM = {
    "enabled": False,  # Set True when Isaac Sim is available
    "host": "localhost",
    "port": 8211,
    "headless": True,
}

# Robot configuration
ROBOT_CONFIG = {
    "type": "sawyer",  # Options: sawyer, franka, ur5
    "simulation": True,  # Use simulation vs real robot
    "workspace_bounds": {
        "x": [-0.5, 0.5],
        "y": [-0.5, 0.5],
        "z": [0.0, 0.8],
    },
}

# =============================================================================
# COST MANAGEMENT
# =============================================================================

# Alert if estimated cost exceeds this threshold (USD)
COST_ALERT_THRESHOLD = 0.10

# Maximum allowed cost per run (USD)
# Set to None for no limit
MAX_COST_PER_RUN = 0.50

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Memory management
MEMORY_CONFIG = {
    # Maximum memory entries to keep in context
    "max_context_entries": 5,
    
    # Enable RAG (future feature)
    "enable_rag": False,
    
    # Memory persistence
    "save_memory": False,
    "memory_file": "agent_memory.json",
}

# Multi-threading (future feature)
CONCURRENCY = {
    "enabled": False,
    "max_workers": 3,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config_summary():
    """Get a summary of current configuration."""
    summary = f"""
Configuration Summary:
=====================
API: {'Enabled' if USE_LLM else 'Disabled (Mock Mode)'}
Model: {CLAUDE_MODEL if USE_LLM else 'N/A'}
Max Iterations: {MAX_ITERATIONS}
Error Threshold: {ERROR_THRESHOLD}
Estimated Cost/Run: {'$0.001-0.005' if 'haiku' in CLAUDE_MODEL.lower() else '$0.01-0.05'}
"""
    return summary


def validate_config():
    """Validate configuration settings."""
    errors = []
    
    if MAX_ITERATIONS < 1:
        errors.append("MAX_ITERATIONS must be >= 1")
    
    if ERROR_THRESHOLD <= 0 or ERROR_THRESHOLD >= 1:
        errors.append("ERROR_THRESHOLD must be between 0 and 1")
    
    if SLINKY_PARAMS["initial_trials"] < 1:
        errors.append("initial_trials must be >= 1")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    return True


# Validate on import
validate_config()
