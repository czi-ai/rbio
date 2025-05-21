# Use PVC folder to load Python environment and code when present.
# This is mostly used in interactive workloads.
PVC_HOME_DIR="${PVC_HOME_DIR:-""}"

# This is where the RBIO code is located
RBIO_HOME_DIR="${RBIO_HOME_DIR:-/work}"

if [ -n "${PVC_HOME_DIR}" ] && [ -d "${PVC_HOME_DIR}" ]; then
    export PVC_HOME_DIR

    # Install packages
    sudo apt-get update
    sudo apt-get install -y clang build-essential vim

    # Setup git completions
    source /usr/share/bash-completion/completions/git

    # Setup vim as default editor
    export EDITOR="vim"
    if ! grep -q "EDITOR=" ~/.bashrc; then
        echo "export EDITOR=${EDITOR}" >> ~/.bashrc
    fi

    if ! grep -q "PVC_HOME_DIR=" ~/.bashrc; then
        echo "export PVC_HOME_DIR=${PVC_HOME_DIR}" >> ~/.bashrc
    fi

    # Link PVC folder to the GCFLow user folder
    ln -s "${PVC_HOME_DIR}" "${HOME}/pvc"

    # Restore the .vscode-server folder
    # This restores workspace settings and extensions
    if [ -d "${PVC_HOME_DIR}/.vscode-server" ]; then
        rm -rf "${HOME}/.vscode-server"
        ln -s "${PVC_HOME_DIR}/.vscode-server" "${HOME}/.vscode-server"
    fi

    # Use cache directory from PVC if present
    # This cache directory is used by UV to cache Python dependencies
    if [ -d "${PVC_HOME_DIR}/.cache" ]; then
        rm -rf "${HOME}/.cache"
        ln -s "${PVC_HOME_DIR}/.cache" "${HOME}/.cache"

        export UV_CACHE_DIR="${PVC_HOME_DIR}/.cache"
        if ! grep -q "UV_CACHE_DIR=" ~/.bashrc; then
            echo "export UV_CACHE_DIR=${PVC_HOME_DIR}/.cache" >> ~/.bashrc
        fi
    fi

    if [ -d "${PVC_HOME_DIR}/pvc/rbio" ]; then
        # Using the RBIO code from PVC
        RBIO_HOME_DIR="${PVC_HOME_DIR}/pvc/rbio"
        cd "${RBIO_HOME_DIR}"

        if [ ! -d "venv" ]; then
            uv venv -n venv --python 3.11 --relocatable
            source venv/bin/activate
            uv pip install poetry
            poetry self add poetry-plugin-export
            poetry export --without-hashes > /tmp/requirements.txt
            uv pip install --quiet -r /tmp/requirements.txt
            uv pip install --no-deps git+https://github.com/czi-ai/transcriptformer
            uv pip install --no-deps .
        else
            source venv/bin/activate
        fi
    fi
else
    # PVC_HOME_DIR is not set or does not exist
    # This is mostly used by training jobs
    cd "${RBIO_HOME_DIR}"
    rm -rf venv
    uv venv -n venv --python 3.11 --relocatable
    source venv/bin/activate
    uv pip install poetry
    poetry self add poetry-plugin-export
    poetry export --without-hashes > /tmp/requirements.txt
    uv pip install --quiet -r /tmp/requirements.txt
    uv pip install --no-deps git+https://github.com/czi-ai/transcriptformer
    uv pip install --no-deps .
fi

export RBIO_HOME_DIR
if ! grep -q "RBIO_HOME_DIR=" ~/.bashrc; then
    echo "export RBIO_HOME_DIR=${RBIO_HOME_DIR}" >> ~/.bashrc
fi

# Generate the accelerate config file
mkdir -p "${HOME}/.cache/huggingface/accelerate/"
ACCELERATE_CONFIG_PATH="${ACCELERATE_CONFIG_PATH:-${HOME}/.cache/huggingface/accelerate/default_config.yaml}"
scripts/generate_accelerate_config.sh > "$ACCELERATE_CONFIG_PATH"

printf "Accelerate config file generated:\n"
echo "$ACCELERATE_CONFIG_PATH"

ln -s "$ACCELERATE_CONFIG_PATH" /work/accelerate_config.yaml

# Setup MLFLow environment variables
export MLFLOW_TRACKING_URI="http://mlflow-api.mlflow.svc.cluster.local:5000"
export MLFLOW_TRACKING_USERNAME="${MLFLOW_TRACKING_USERNAME:-${GCFLOW_ENV_MAIL}}"
export MLFLOW_TRACKING_PASSWORD="${MLFLOW_TRACKING_PASSWORD:-${MLFLOW_TOKEN}}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-rbio}"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING="${MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING:-true}"
if ! grep -q "MLFLOW_TRACKING_URI=" ~/.bashrc; then
    echo "export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}" >> ~/.bashrc
fi
if ! grep -q "MLFLOW_TRACKING_USERNAME=" ~/.bashrc; then
    echo "export MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}" >> ~/.bashrc
fi
if ! grep -q "MLFLOW_TRACKING_PASSWORD=" ~/.bashrc; then
    echo "export MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}" >> ~/.bashrc
fi
if ! grep -q "MLFLOW_EXPERIMENT_NAME=" ~/.bashrc; then
    echo "export MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME}" >> ~/.bashrc
fi
if ! grep -q "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=" ~/.bashrc; then
    echo "export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=${MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING}" >> ~/.bashrc
fi

sudo chown -R flow:flow /mnt/czi-sci-ai/project-rbio-large/checkpoints/
sudo chmod -R 777 /mnt/czi-sci-ai/project-rbio-large/checkpoints/
