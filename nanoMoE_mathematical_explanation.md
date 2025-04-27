# Mathematical Explanation of nanoMoE Architecture

## 1. Overview of Mixture-of-Experts (MoE)

The Mixture-of-Experts (MoE) architecture is a neural network design that employs multiple specialized "expert" networks, each focusing on different aspects of the input data. In the context of language models, MoE replaces some of the standard feed-forward networks (FFNs) in transformer blocks with expert layers.

## 2. Key Components and Mathematical Formulations

### 2.1 Standard Transformer Block

In a standard transformer block, the computation flow is:

1. **Self-Attention**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

2. **Feed-Forward Network**:
   $$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

3. **Residual Connections and Layer Normalization**:
   $$\text{Block}(x) = \text{LayerNorm}(x + \text{Attention}(x))$$
   $$\text{Output} = \text{LayerNorm}(\text{Block}(x) + \text{FFN}(\text{Block}(x)))$$

### 2.2 MoE Transformer Block

In an MoE transformer block, the FFN is replaced with an MoE layer:

$$\text{Output} = \text{LayerNorm}(\text{Block}(x) + \text{MoE}(\text{Block}(x)))$$

### 2.3 Router Mechanism

The router determines which experts should process each token:

1. **Router Logits Computation**:
   $$\text{logits} = xW_g$$

2. **Optional Noise Addition**:
   $$\text{noise} = \text{softplus}(xW_{\text{noise}}) \cdot \mathcal{N}(0, 1)$$
   $$\text{logits}_{\text{noisy}} = \text{logits} + \text{noise}$$

3. **Top-K Selection**:
   Select the $k$ highest values from $\text{logits}$ for each token.

4. **Expert Probabilities**:
   $$p(e|x) = \frac{\exp(\text{logits}_e)}{\sum_{i \in \text{top-k}} \exp(\text{logits}_i)}$$

### 2.4 Expert Capacity and Token Routing

The expert capacity determines how many tokens can be processed by each expert:

$$\text{capacity} = \left\lfloor \frac{k \cdot \text{capacity\_factor} \cdot \text{num\_tokens}}{n_{\text{experts}}} \right\rfloor$$

Tokens exceeding the capacity are dropped and passed through the residual connection.

### 2.5 Expert Processing

Each expert is a feed-forward network:

$$\text{Expert}_i(x) = \text{GELU}(xW_{i,1} + b_{i,1})W_{i,2} + b_{i,2}$$

### 2.6 Combining Expert Outputs

The final output is a weighted combination of expert outputs:

$$\text{MoE}(x) = \sum_{i \in \text{top-k}} p(i|x) \cdot \text{Expert}_i(x)$$

## 3. Auxiliary Losses

### 3.1 Load Balancing Loss

This loss encourages uniform expert utilization:

1. **Expert Importance** - Fraction of router probability allocated to each expert:
   $$P_i = \frac{1}{N} \sum_{j=1}^{N} p(i|x_j)$$

2. **Expert Load** - Fraction of tokens dispatched to each expert:
   $$L_i = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}[i \in \text{top-k}(x_j)]$$

3. **Load Balancing Loss**:
   $$\mathcal{L}_{\text{balance}} = n_{\text{experts}} \cdot \sum_{i=1}^{n_{\text{experts}}} P_i \cdot L_i$$

### 3.2 Router Z-Loss

This loss constrains the magnitude of router logits:

$$\mathcal{L}_{\text{z}} = \frac{1}{N} \sum_{j=1}^{N} \left(\log \sum_{i=1}^{n_{\text{experts}}} \exp(\text{logits}_{i,j})\right)^2$$

### 3.3 Combined Loss

The final loss combines the standard cross-entropy loss with the auxiliary losses:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \alpha \cdot \mathcal{L}_{\text{balance}} + \beta \cdot \mathcal{L}_{\text{z}}$$

where $\alpha$ and $\beta$ are scaling factors (typically 0.001 and 0.01, respectively).

## 4. Implementation Details

### 4.1 Expert Layer Implementation

In nanoMoE, experts are implemented efficiently using batch matrix multiplication:

```python
# Forward pass through all experts in batch
x = torch.bmm(x, self.c_fc)  # [n_exp, batch_size, 4*d]
x = self.gelu(x)
x = torch.bmm(x, self.c_proj)  # [n_exp, batch_size, d]
```

### 4.2 MoE Stride

MoE models typically use a stride of $P$, meaning every $P$-th transformer layer is converted to an MoE layer:

```python
transformer_blocks = []
for i in range(num_blocks):
    use_moe = (i % P) == 0
    transformer_blocks.append(Block(use_moe=use_moe))
```

## 5. Advantages of MoE Architecture

1. **Parameter Efficiency**: MoE models can scale to trillions of parameters while keeping the computational cost manageable.

2. **Conditional Computation**: Only a subset of parameters (active experts) is used for each token.

3. **Specialization**: Different experts can specialize in different aspects of language processing.

4. **Scaling Law Improvements**: MoE models often show better performance scaling with parameter count compared to dense models.