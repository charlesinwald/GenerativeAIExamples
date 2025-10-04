# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a Streamlit-based data analysis agent that uses NVIDIA's Llama-3.1-Nemotron-Ultra and Llama-3.3-Nemotron-Super models for interactive data exploration. The application implements an agentic architecture with specialized components for query understanding, code generation, execution, and reasoning.

## Key Architecture Components

### Agent System Architecture
The application uses a multi-agent system with distinct roles:
- **QueryUnderstandingTool**: Classifies user queries to determine if visualization is needed
- **CodeGenerationAgent**: Generates pandas/matplotlib code based on user queries and data schema
- **ExecutionAgent**: Safely executes generated code in a controlled environment
- **ReasoningAgent**: Provides explanations and insights about results with model "thinking" streams
- **DataInsightAgent**: Generates initial dataset summaries and suggested questions

### Model Configuration System
The app supports multiple models through a `ModelConfig` class that defines model-specific parameters:
- Different temperature and max_token settings per agent type
- Model-specific reasoning prompts (`reasoning_false` for concise mode, `reasoning_true` for detailed thinking)
- Currently supports two NVIDIA models with different parameter optimization

### Code Generation Strategy
- **Dual-path code generation**: Separate tools for data analysis vs visualization
- **Context-aware prompts**: Uses conversation history to resolve ambiguous references
- **Pandas-only constraint**: Generated code is restricted to pandas operations and matplotlib for safety

## Common Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Set NVIDIA API key
export NVIDIA_API_KEY=your_api_key_here

# Run the Streamlit app
streamlit run data_analysis_agent.py
```

### Development and Testing
```bash
# Download sample dataset for testing
wget https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# Test with different models by setting environment variable
export DEFAULT_MODEL=llama-3-3-nemotron-super-v1-5
streamlit run data_analysis_agent.py
```

### Code Structure
- **Single-file application**: All functionality contained in `data_analysis_agent.py`
- **Session state management**: Streamlit session state handles data persistence, chat history, and model configuration
- **Streaming responses**: ReasoningAgent uses streaming to display model "thinking" in real-time

## Development Notes

### Adding New Models
To add support for new models:
1. Add model configuration to `MODEL_CONFIGS` dictionary in the format:
   ```python
   "model-key": ModelConfig(
       model_name="nvidia/model-name",
       model_url="https://build.nvidia.com/model-link",
       model_print_name="Display Name",
       # Set temperatures and max_tokens for each agent
       reasoning_false="instruction for concise mode",
       reasoning_true="instruction for detailed thinking mode"
   )
   ```
2. Update the model selector in the main UI

### Code Generation Constraints
- All generated code must use pandas operations on the `df` variable
- Results must be assigned to a `result` variable
- No file I/O, network requests, or dangerous operations allowed
- Visualization code must use matplotlib with specific figure size constraints

### Session State Variables
Key session state variables:
- `df`: Current loaded DataFrame
- `messages`: Chat history with role/content/plot_index structure
- `plots`: List of matplotlib figures for display
- `current_model`: Selected model configuration key
- `insights`: Generated dataset summary from DataInsightAgent

### Error Handling Pattern
The application uses a consistent error handling pattern where execution errors are returned as strings starting with "Error executing code:" and processed by the ReasoningAgent for user-friendly explanations.
