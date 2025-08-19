uv venv -n venv --python 3.11 --relocatable
source venv/bin/activate
uv sync --active
cp /mnt/czi-sci-ai/project-rbio-80t/rbio-public/training-files/* training/
cd training