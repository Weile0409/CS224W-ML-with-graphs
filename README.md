# CS224W: Machine Learning with Graphs
### My self-study note and solutions
### Link: http://web.stanford.edu/class/cs224w/

- Colab 0: NetworkX Tutorial, PyTorch Geometric Tutorial, Implementing Graph Neural Networks (GNNs)


## Notes
- Lecture 1: Introduction
  - Introduce application and use cases of ML in graphs
  - Introduce different types of tasks: Node level, edge level and graph level
  - Introduce choice of a graph representation: Undirected, bipartite, weighted, adjacency matrix, etc
- Lecture 2: Feature Engineering for ML in graphs
  - Traditional ML pipeline: hand-crafted (structural) features + ML models (eg. SVM)
    - Node-level features: Node degree, centrality, clustering coefficient, graphlets
    - Link-level features: Distance-based feature, local/global neighborhood overlap
    - Graph-level features: Graphlet kernel, WL kernel
  - We only considered featurizing the graph structure (but not the attribute of nodes and their neighbors)
