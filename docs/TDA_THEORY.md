# Theoretical Background: Algebraic Topology in Finance

This document explains the mathematical foundations behind the **TDA Trading Agent**. We utilize **Topological Data Analysis (TDA)** to extract shape-based features from financial time series, which act as robust inputs for our Reinforcement Learning model.

## 1. Time Delay Embedding (Takens' Theorem)

Financial data typically comes as a single 1D time series (e.g., Closing Prices). However, the underlying market dynamics are complex and multidimensional.

**Takens' Theorem** allows us to reconstruct the "phase space" (the hidden state space) of the dynamic system from a single observable variable.

Given a time series $X = [x_1, x_2, ..., x_N]$, we create vectors:

$$ V_i = [x_i, x_{i+\tau}, x_{i+2\tau}, ..., x_{i+(m-1)\tau}] $$

Where:
*   $m$ is the **Embedding Dimension** (e.g., 3).
*   $\tau$ is the **Time Delay** (e.g., 1).

This transforms our 50-day window of prices into a **Point Cloud** in 3D space. The shape of this point cloud (manifold) characterizes the market's behavior during that window.

## 2. Persistent Homology

Once we have a point cloud, we want to quantify its "shape". We look for topological features that persist across different scales.

### Homology Groups
*   **$H_0$ (Connected Components)**: Represents clusters of data points. In financial terms, this can indicate regimes of volatility or stability.
*   **$H_1$ (Loops/Cycles)**: Represents "holes" in the 1D structure. In phase space, loops often correspond to periodic or recurring behaviors in the market (e.g., mean reversion cycles).
*   **$H_2$ (Voids)**: Trapped volumes (used in higher dimensions).

### Filtration & Persistence
We cannot just connect points at a fixed radius because the "correct" scale is unknown. Instead, we use a **Filtration**:
1.  Grow balls of radius $r$ around every point.
2.  Connect points when balls overlap (forming a Simplicial Complex, specifically a Vietoris-Rips complex).
3.  Track when topological features (components, loops) are **Born** and when they **Die** (get filled in) as $r$ increases.

This yields a **Persistence Diagram**, a plot of (Birth, Death) pairs for each feature.

## 3. Financial Interpretation

*   **Long-lived $H_0$ features**: Indicate that the data points are spread out and distinct (high volatility, sparse sampling).
*   **Long-lived $H_1$ features (Loops)**: Indicate strong cyclic, non-linear structure. This is often a signal of a "market attractor" or a specific repetitive pattern that linear models might miss.
*   **Noise**: Short-lived features (close to the diagonal on the diagram) are typically considered noise.

## 4. Vectorization for Machine Learning

Neural networks cannot directly ingest Persistence Diagrams (sets of points). We must vectorizes them.

### Persistent Entropy
We calculate the Shannon Entropy of the lifetimes of the features.
$$ E = -\sum p_i \log(p_i) $$
Where $p_i$ is the relative lifetime of feature $i$.
*   **High Entropy**: Complex, diverse topological features.
*   **Low Entropy**: Simple structure (e.g., a single straight line or cluster).

### Statistics
We also use:
*   **Mean Lifetime**: Average persistence of features.
*   **Max Lifetime**: The most dominant feature's persistence (e.g., the "main" loop).

These 6 scalar values (Entropy, Mean, Max for both $H_0$ and $H_1$) form the input vector for our RL Agent.
