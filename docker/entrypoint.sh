# docker/entrypoint.sh (CONFIRMED FROM YOUR LAST INPUT - GOOD)
#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Conda environment activation ---
# The CONDA_DIR should be /opt/conda as set in Dockerfile
CONDA_BASE_PATH="/opt/conda" # Default install path for Miniconda/Anaconda
PROFILE_SCRIPT="${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
ENV_NAME="asys-i" # Name of the conda environment

if [ -f "$PROFILE_SCRIPT" ]; then
    # shellcheck source=/dev/null
    source "$PROFILE_SCRIPT"
    echo "Sourced conda profile script from $PROFILE_SCRIPT"
    
    # Check if environment exists before trying to activate
    if conda env list | grep -q "${ENV_NAME}"; then
        echo "Activating Conda environment: ${ENV_NAME}..."
        conda activate "${ENV_NAME}"
        if [ "$CONDA_DEFAULT_ENV" = "${ENV_NAME}" ]; then
            echo "Successfully activated Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
        else
            echo "Error: Failed to activate Conda environment '${ENV_NAME}'. CONDA_DEFAULT_ENV is '$CONDA_DEFAULT_ENV'."
            # Optionally, list available environments for debugging
            echo "Available environments:"
            conda env list
            exit 1 # Exit if activation fails
        fi
    else
        echo "Error: Conda environment '${ENV_NAME}' not found."
        echo "Available environments:"
        conda env list
        exit 1 # Exit if environment not found
    fi
else
    echo "Error: Conda profile script not found at $PROFILE_SCRIPT. Conda environment cannot be activated."
    exit 1 # Exit if conda itself is not setup correctly
fi

# --- Execute the passed command ---
echo "Executing command with UID $(id -u) GID $(id -g): $@"
exec "$@"
