# A Unified Spectral-Spatial Framework for GNNs: Balancing Over-Smoothing and Over-Squashing

## abstract
Over-smoothing (OSM) and over-squashing (OSQ) are two fundamental phenomena that limit the performance of Graph Neural Networks (GNNs), yet a unified spectral-spatial understanding of these phenomena remains underexplored. In this paper, we adopt polynomial spectral filters as an analytical tool to establish a unified spectral-spatial framework for graph convolution and systematically characterize the effect of the polynomial order \(k\) on GNN performance. Within this framework, we reveal an intrinsic trade-off induced by the polynomial order. Specifically, higher-order filters enhance spectral expressiveness and alleviate OSM caused by the dominant low-frequency components. However, they also expand the spatial receptive field, thereby intensifying information compression and increasing the risk of OSQ. Based on this analysis, we derive a principled criterion for balancing the polynomial order and propose a Quadratic Spectral Graph Convolution Network (QS-GCN) for graph classification. Experiments demonstrate the effectiveness and robustness of the proposed method.
## Prerequisites

- pytorch 2.4.1+cu124
- torch_geometric 2.6.1

## Running the Project

1. **Run `main.py` for model training and testing:**
   ```bash
   python main.py