# docker/Dockerfile (REVISED FOR NON-ROOT USER & C++ BUILD)

# --- Builder Stage for C++ dependencies and Python base ---
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04 AS builder

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV CONDA_DIR /opt/conda

# Install essential build tools, Boost, CMake, Ninja, and Python dev tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    # Boost development libraries are crucial for the C++ core
    libboost-dev libboost-system-dev libboost-filesystem-dev libboost-program-options-dev libboost-interprocess-dev \
    # Python build dependencies
    libffi-dev libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev xz-utils tk-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -o ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh && \
    ${CONDA_DIR}/bin/conda init bash

# Argument for selecting profile specific dependencies (e.g., 'hpc' or 'all')
ARG BUILD_PROFILE_EXTRA=all

# Copy project files needed for dependency installation and C++ build
WORKDIR /build_tmp
COPY pyproject.toml poetry.lock* README.md LICENSE ./
# Copy the C++ source code needed for compilation
COPY src/asys_i/hpc ./src/asys_i/hpc
# If other C++ sources exist outside hpc/, copy them too. For example:
# COPY third_party ./third_party # If moodycamel or other submodules are here

# Create Conda environment and install Python dependencies + C++ extension
# The `pip install .[${BUILD_PROFILE_EXTRA}]` will trigger scikit-build-core
SHELL ["/bin/bash", "-c"]
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda create -n asys-i python=${PYTHON_VERSION} -y && \
    conda activate asys-i && \
    conda install pip -y && \
    # Install project with selected extras, this builds the C++ extension
    echo "Attempting to build with profile extra: ${BUILD_PROFILE_EXTRA}" && \
    pip install --no-cache-dir ."[${BUILD_PROFILE_EXTRA}]" && \
    # Clean up build artifacts from /build_tmp if any large ones remain
    # (scikit-build-core usually builds in a temporary dir)
    conda clean -afy && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete


# --- Final Runtime Stage ---
FROM nvidia/cuda:12.1.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR /opt/conda
ENV PATH=${CONDA_DIR}/envs/asys-i/bin:${CONDA_DIR}/bin:${PATH}
ENV PYTHONNOUSERSITE=1
ENV LD_LIBRARY_PATH=${CONDA_DIR}/envs/asys-i/lib:${LD_LIBRARY_PATH} # Ensure C++ shared libs are found

# Create a non-root user and group
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} --system asys_i_group && \
    useradd -u ${UID} -g asys_i_group -ms /bin/bash --system asys_i_user

# Copy the built Conda environment from the builder stage
COPY --from=builder ${CONDA_DIR} ${CONDA_DIR}

# Copy the rest of the application code
WORKDIR /home/asys_i_user/app
COPY . .
# Ensure the app directory is owned by the non-root user
RUN chown -R asys_i_user:asys_i_group /home/asys_i_user ${CONDA_DIR} && \
    # Ensure conda env is writable by user for any runtime linking if needed (though ideally not)
    # More securely, make only specific dirs writable if necessary.
    chmod -R g+w ${CONDA_DIR}/envs/asys-i

# Switch to the non-root user
USER asys_i_user

# Expose ports (Prometheus, TensorBoard etc.)
EXPOSE 8001 6006 3000 9090

# Healthcheck (optional, adjust to your app's health endpoint)
# HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
#   CMD curl -f http://localhost:8001/metrics || exit 1

# Default entrypoint script
ENTRYPOINT ["/home/asys_i_user/app/docker/entrypoint.sh"]

# Default command (can be overridden in docker-compose.yml)
CMD ["asys-i-run"]
