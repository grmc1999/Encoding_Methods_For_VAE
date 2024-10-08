FROM tverous/pytorch-notebook:latest

RUN python -m pip install einops
RUN python -m pip install pyro-ppl
RUN python -m pip install

#INSTALL PYTHON GEOMETRY
RUN export TORCH=$(python -c "import torch; print(torch.__version__)")

