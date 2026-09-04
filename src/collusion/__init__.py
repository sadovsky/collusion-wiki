"""Graph-structural analysis of the OpenAI agent wiki collusion corpus.

The corpus published at https://collusion.wiki/ is a flat event log. Nothing in
it is a graph. This package reconstructs the latent network structure -- who
handed off to whom, which pages link to which, which agents adopted which
bypass technique -- and measures it.
"""

__version__ = "0.1.0"

DATA_DIR = "data"
DERIVED_DIR = "derived"
FIGURES_DIR = "figures"
