# Computing linear sections of varieties
Efficient implementation of the algorithm of Johnston, Lovitz and Vijayaraghavan (see [paper](https://arxiv.org/abs/2212.03851)) for finding the intersection of a linear subspace and a conic variety in Python.

### Current features
- Compute $(\mathcal W, \mathcal X)$-decompositions for $\mathcal X$ any Segre-Veronese variety (space of rank-1 tensors with partial symmtries along any number of modes).
- Partial tensor decomposition method via Jennrich's algorithm on flattening of arbitrary tensor to 3rd order. Including
  - modifications to original simultaneous diagonalization for the decomposition of complex tensors,
  - greedy matching for eigenvalues sets in simultaneous diagonalization for robustness when decomposing noisy tensors,
  - discrepancy and approximation error reporting at multiple levels,
  - decomposition error methods to identify overcomplete tensors or problems with decomposition process.
- Helper functions for symmetric lifts/symmetric khatri rhao products of subspaces.
- Helper functions for partial expansions of symmetric tensors (Veronese embeddings to Segre embeddings along specified modes).
- Efficient generation of an a basis of polynomials for the degree-2 component of the ideal for a Segre-Veronese variety.
