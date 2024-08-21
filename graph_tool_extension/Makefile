CONDA_ENV_NAME=gt

# Automatically detect Python version
PYTHON_VERSION=$(shell python -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))")

CONDA_ENV_PATH=$(HOME)/.conda/envs/$(CONDA_ENV_NAME)
PKG_CONFIG_PATH=$(CONDA_ENV_PATH)/lib/pkgconfig
GRAPH_TOOL_PKG=graph-tool-py$(PYTHON_VERSION).pc

CXX=g++
CXXFLAGS=-O3 -fopenmp -std=gnu++17 -Wall -fPIC `pkg-config --cflags --libs $(PKG_CONFIG_PATH)/$(GRAPH_TOOL_PKG)` -shared

ALL: libshared_mcmc.so

libshared_mcmc.so: shared_blockmodel_mcmc.hh shared_blockmodel_mcmc.cc from_python_conversion.hh
	$(CXX) $(CXXFLAGS) shared_blockmodel_mcmc.cc -o $@