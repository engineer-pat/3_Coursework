# Differential Equation Solver - Streamlit App

This Streamlit app solves and visualizes exponential decay differential equations.

## Setup and Run

### First-time setup
1. Make sure you're in the project directory
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. If needed, install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the app
1. Activate the virtual environment (if not already activated):
   ```bash
   source venv/bin/activate
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Features
- Solves the exponential decay differential equation dy/dt = -k*y
- Compares numerical solution with analytical solution
- Interactive parameters:
  - Initial value (y₀)
  - Decay constant (k)
  - Time range
  - Number of points for plotting 