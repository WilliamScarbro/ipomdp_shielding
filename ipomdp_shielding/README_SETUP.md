# Setup Instructions for ipomdp_shielding

This guide explains how to set up the development environment for the ipomdp_shielding project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Quick Setup

### Option 1: Using the setup script (recommended)

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

### Option 2: Manual setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Activating the Environment

After setup, you need to activate the virtual environment before working on the project:

```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt, indicating the environment is active.

## Deactivating the Environment

When you're done working:

```bash
deactivate
```

## Running the Project

With the virtual environment activated, you can run project modules. For example:

```bash
# Run Monte Carlo experiments
python -m ipomdp_shielding.MonteCarlo.experiments

# Run belief selector tests
python -m ipomdp_shielding.MonteCarlo.test_belief_selector
```

## Dependencies

The project uses the following main dependencies:

- **numpy**: Numerical computing and array operations
- **scipy**: Scientific computing (optimization, spatial algorithms)
- **statsmodels**: Statistical models and confidence intervals
- **matplotlib**: Plotting and visualization

All dependencies are specified in `requirements.txt`.

## Troubleshooting

### Virtual environment not activating

Make sure you're using the correct command for your shell:
- Bash/Zsh: `source venv/bin/activate`
- Fish: `source venv/bin/activate.fish`
- Windows (cmd): `venv\Scripts\activate.bat`
- Windows (PowerShell): `venv\Scripts\Activate.ps1`

### Import errors

Make sure:
1. The virtual environment is activated
2. All dependencies are installed: `pip install -r requirements.txt`
3. You're running Python from the project root directory

### Permission denied on setup.sh

Run: `chmod +x setup.sh` to make the script executable.

## Verifying Installation

To verify everything is set up correctly:

```bash
# Activate the environment
source venv/bin/activate

# Check installed packages
pip list

# Try importing the main modules
python -c "import ipomdp_shielding; import numpy; import scipy; print('All imports successful!')"
```

## Development

When adding new dependencies:

1. Install the package: `pip install package_name`
2. Update requirements.txt: `pip freeze > requirements.txt`
3. Commit both changes to version control

Or manually add the package to `requirements.txt` with version constraints.
