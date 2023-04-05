# CS224W: Machine Learning with Graphs
### My self-study note and solutions
### Link: http://web.stanford.edu/class/cs224w/

- Colab 0: NetworkX Tutorial, PyTorch Geometric Tutorial, Implementing Graph Neural Networks (GNNs)


## Notes
### Lecture 1: Introduction
- Introduce application and use cases of ML in graphs
- Introduce different types of tasks: Node level, edge level and graph level
- Introduce choice of a graph representation: Undirected, bipartite, weighted, adjacency matrix, etc

### Lecture 2: Feature Engineering for ML in graphs
- Traditional ML pipeline: hand-crafted (structural) features + ML models (eg. SVM)
  - Node-level features: Node degree, centrality, clustering coefficient, graphlets
  - Link-level features: Distance-based feature, local/global neighborhood overlap
  - Graph-level features: Graphlet kernel, WL kernel
- We only considered featurizing the graph structure (but not the attribute of nodes and their neighbors

### Lecture 3: Node Embeddings
- Before: Given an input graph, extract node, link and graph-level features, learn a model (SVM, neural network, etc.) that maps features to labels.
- Now: Graph Representation Learning alleviates the need to do feature engineering every single time.
- Discussed graph representation learning, a way to learn node and graph embeddings for downstream tasks , without feature engineering
  - Idea: Map nodes into an embedding space such that similarity of embeddings between nodes indicates their similarity in the network
  - Downstream tasks: node/graph classification, link prediction, anomalous node detection, etc
- Diccussed Encoder-decoder framework, Node similarity measure ((biased) random walk) and Extension to Graph embedding
- Dicussed how to use trained embeddings of nodes
- Limitations of node embeddings via matrix factorization and random walks
  - Cannot obtain embeddings for nodes not in the training set
  - Cannot capture structural similarity
  - Cannot utilize node, edge and graph features

### Lecture 4: Graph Neural Networks
- Introduce Graph Convolutional Network
- CNN can be viewed as a special GNN

### Lecture 5: A General Perspective on GNNs
- GNN layer: Transformation and Aggregation, introduce some classical GNN layers
- Layer connectivity: Designing # of layers, skip connections
- Graph manipulation
  - Feature augmentation
  - Structure manipulation: Solve sparse graph issue(add virtual nodes/edges), node neighborhood sampling if too many neighbors
  
 ### Lecture 6: GNN Augmentation and Training
 - Introduce a general GNN framework 
  - Lecture 5: GNN layer, layer connectivity, Graph augmentation
  - Lecture 6: Learning objectives (The full training pipeline of a GNN)

 ### Lecture 7: Theory of Graph Neural Networks
