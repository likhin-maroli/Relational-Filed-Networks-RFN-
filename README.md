# 🧠 Relational Field Network (RFN)

> A novel neural architecture for learning pairwise feature interactions — no graphs, no sequences, just pure relational learning over vector inputs.

---

## 📌 What is RFN?

**Relational Field Network (RFN)** is a new type of neural network that learns how **input features influence each other** — directly modeling **relational dependencies between features** rather than treating them as independent or flattened.

Unlike Transformers (which attend across tokens) or GNNs (which rely on graph structures), RFN:
- Works on plain tabular/vector data
- Does **self-attention between features** within each input
- Learns interpretable **relational maps** from scratch
- Can be plugged in as a drop-in replacement for MLPs

---

## 🔧 Architecture Overview

Each RFN layer:
1. Projects each input feature into query & key spaces
2. Computes pairwise relational scores between features
3. Uses softmax-weighted aggregation to update feature values

This mechanism resembles self-attention — but is applied **within each input vector**, not across a sequence.

```python
class RelationalFieldLayer(nn.Module):
    def forward(self, x):
        # x: (batch_size, num_features, feature_dim)
        ...
