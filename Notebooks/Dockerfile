FROM tverous/pytorch-notebook:latest

RUN python -m pip install einops
RUN python -m pip install pyro-ppl
RUN python -m pip install matplotlib
#!pip install ipywidgets widgetsnbextension pandas-profiling

#INSTALL PYTHON GEOMETRY
RUN export TORCH=$(python -c "import torch; print(torch.__version__)")

RUN python -m pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
RUN python -m pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
RUN python -m pip install -q git+https://github.com/pyg-team/pytorch_geometric.git