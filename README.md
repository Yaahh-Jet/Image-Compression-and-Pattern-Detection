# Image Compression and Pattern Detection Using Eigenvalues and Orthogonal Projections

**Linear Algebra Mini Project — PES University**

---

## Overview

This project implements two core applications of Linear Algebra on image data:

1. **SVD-Based Image Compression** — Represent an image using only the top-k singular values, reducing storage while preserving visual quality.
2. **Eigenfaces for Pattern Detection** — Use PCA (via eigendecomposition of the covariance matrix) to find dominant face patterns across a dataset, then reconstruct faces as orthogonal projections onto the eigenface subspace.

---

## Project Structure

```
eigenfaces_project/
│
├── eigenfaces_project.ipynb   ← Main notebook (all code + explanations)
│
├── README.md                  ← This file
│
└── outputs/                   ← Generated plots
    ├── svd_compression.png    ← Compression at different k values + error maps
    ├── svd_analysis.png       ← Singular value spectrum + cumulative energy
    ├── eigenfaces.png         ← Mean face + top eigenfaces visualised
    ├── face_reconstruction.png← Face reconstruction via orthogonal projection
    └── pca_analysis.png       ← Variance explained + reconstruction error vs k
```

---

## Setup

### Requirements

```bash
pip install numpy matplotlib scikit-learn scipy jupyter
```

### Dataset

The notebook uses the **Olivetti Faces Dataset** — 400 grayscale images (64×64) of 40 people.

```python
from sklearn.datasets import fetch_olivetti_faces
data = fetch_olivetti_faces(shuffle=True, random_state=42)
faces = data.images  # shape: (400, 64, 64)
```

> If you're offline, the notebook automatically falls back to synthetic low-rank data with the same shape and API.

### Run

```bash
jupyter notebook eigenfaces_project.ipynb
```

---

## Key Concepts

### SVD-Based Image Compression

Any image matrix $A \in \mathbb{R}^{m \times n}$ decomposes as:

$$A = U \Sigma V^T = \sum_{i=1}^{r} \sigma_i \, u_i v_i^T$$

The **rank-k approximation** keeps only the top-k terms:

$$A_k = \sum_{i=1}^{k} \sigma_i \, u_i v_i^T$$

By the **Eckart–Young theorem**, $A_k$ is the best rank-k approximation in Frobenius norm. The compression ratio is:

$$\text{ratio} = \frac{k(m + n + 1)}{mn}$$

### Eigenfaces via PCA

Given N face images flattened to vectors $x_i \in \mathbb{R}^d$:

1. Compute mean face: $\bar{x} = \frac{1}{N}\sum x_i$
2. Center data: $\tilde{x}_i = x_i - \bar{x}$
3. SVD of centered data matrix $X$: right singular vectors = **eigenfaces**
4. Eigenvalues of covariance matrix: $\lambda_i = \sigma_i^2 / N$

### Orthogonal Projection

Reconstruct a face using top-k eigenfaces:

$$\hat{x} = \bar{x} + Q_k Q_k^T (x - \bar{x})$$

where $Q_k Q_k^T$ is the **orthogonal projection matrix** onto the k-dimensional eigenface subspace.

---

## Notebook Sections

| Section | What it does |
|---|---|
| 1 | Imports + dataset loading |
| 2 | Visualise sample images |
| 3 | SVD compression + error/ratio analysis |
| 4 | PCA eigenface computation |
| 5 | Face reconstruction via orthogonal projection |
| 6 | Variance explained + error vs k plots |
| 7 | Verify SVD ↔ eigendecomposition equivalence |
| 8 | Summary table + takeaways |

---

## Results Summary

- At **k=10** (15.6% storage), SVD reconstructs images with >90% energy preserved
- Top **20 eigenfaces** capture ~80% of total variance across the dataset
- Orthogonal projection with **k=50 eigenfaces** gives near-perfect face reconstruction
- SVD of data matrix $X$ and eigendecomposition of $X^TX$ give **identical eigenvalues** (verified numerically)

---

## Real-World Applications

| This project covers... | Used in real life for... |
|---|---|
| SVD compression | JPEG-like image compression, video streaming |
| Eigenfaces / PCA | Face recognition, biometric systems |
| Low-rank approximation | Recommendation systems (Netflix, Spotify) |
| Covariance eigendecomposition | Medical imaging (MRI/CT compression) |

---

## References

- Strang, G. — *Linear Algebra and Its Applications*
- Turk & Pentland (1991) — *Eigenfaces for Recognition*
- Olivetti Faces Dataset — AT&T Laboratories Cambridge
- `sklearn.datasets.fetch_olivetti_faces` — [scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html)
