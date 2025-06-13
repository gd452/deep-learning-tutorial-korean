# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a step-by-step deep learning tutorial in Korean, designed to teach fundamental concepts before diving into Large Language Models (LLMs). The tutorials are organized as Jupyter notebooks with detailed explanations and hands-on code examples.

## Project Structure

```
tiny-llm-by-claude/
├── tutorials/
│   ├── step1_basics/         # Python and NumPy fundamentals
│   ├── step2_perceptron/     # Basic neural network concepts
│   ├── step3_mlp/           # Multi-layer perceptron from scratch
│   ├── step4_pytorch/       # PyTorch framework introduction
│   ├── step5_cnn/          # Convolutional Neural Networks
│   ├── step6_rnn/          # Recurrent Neural Networks
│   ├── data/               # Downloaded datasets (MNIST, CIFAR-10)
│   └── utils/              # Helper functions and utilities
├── README.md               # Project overview and learning roadmap
├── requirements.txt        # Python dependencies
└── CLAUDE.md              # This file
```

## Key Commands

### Setup and Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Running Individual Notebooks
```bash
# Run a specific notebook
jupyter notebook tutorials/step1_basics/01_python_numpy_basics.ipynb
```

### Testing Code Examples
```bash
# Convert notebook to Python script and run
jupyter nbconvert --to python tutorials/step1_basics/01_python_numpy_basics.ipynb
python tutorials/step1_basics/01_python_numpy_basics.py
```

## Code Architecture

### Tutorial Structure
Each tutorial notebook follows a consistent structure:
1. **Introduction**: Learning objectives and prerequisites
2. **Theory**: Conceptual explanations with visualizations
3. **Implementation**: Step-by-step code implementation
4. **Experiments**: Hands-on exercises with different parameters
5. **Exercises**: Practice problems for reinforcement

### Key Design Patterns
- **Progressive Complexity**: Each step builds on previous concepts
- **From Scratch First**: Implement algorithms in NumPy before using frameworks
- **Visual Learning**: Extensive use of matplotlib for visualizations
- **Practical Examples**: Real-world applications in each tutorial

### Language Considerations
- All explanations and comments are in Korean
- Variable names and function names use English for compatibility
- Error messages and system outputs remain in English

## Development Guidelines

When modifying or extending tutorials:
1. Maintain the step-by-step approach with clear explanations
2. Include visualizations for complex concepts
3. Provide both NumPy and PyTorch implementations where applicable
4. Test all code cells to ensure they run without errors
5. Keep Korean language consistency in markdown cells