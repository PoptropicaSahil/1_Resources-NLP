# Notes from the blog post: Einsum is All you Need - Einstein Summation in Deep Learning <https://rockt.ai/2018/04/30/einsum>

> Tim Rocktäschel, 30/04/2018 – updated 02/05/2018

## Matrix Transpose

$$ B_{ji}=A_{ij} $$

```python
import torch
a = torch.arange(6).reshape(2, 3)
torch.einsum('ij->ji', [a])
tensor([[ 0.,  3.],
        [ 1.,  4.],
        [ 2.,  5.]])
```

## Matrix Sum

$$ b = ∑_i∑_j A_{ij}=A_{ij} $$

```py
a = torch.arange(6).reshape(2, 3)
torch.einsum('ij->', [a])
tensor(15.)
```

## Column Sum

$$ b_j=∑_i A_{ij}=A_{ij} $$
> Only the column remains!

```py
a = torch.arange(6).reshape(2, 3)
torch.einsum('ij->j', [a])
tensor([ 3.,  5.,  7.])
```

## Row Sum

$$ b_i=∑_j A_{ij} = A_{ij} $$

```py
a = torch.arange(6).reshape(2, 3)
torch.einsum('ij->i', [a])
tensor([  3.,  12.])
```

## Matrix-Vector Multiplication

$$ c_i = ∑_k A_{ik} b_k = A_{ik} b_k $$

```py
a = torch.arange(6).reshape(2, 3)
b = torch.arange(3)
torch.einsum('ik,k->i', [a, b])
tensor([  5.,  14.])
```

> **NOTE:** The result is correctly shaped as **(2,) (a 1D tensor) rather than (2, 1) (a 2D column vector) because einsum automatically squeezes singleton dimensions unless explicitly specified in the output notation.** A matrix with shape $(2, 3)$ multiplied by a vector with shape $(3,)$ produces a **1D output of shape $(2,)$ in standard linear algebra.** The einsum notation `ik,k->i` explicitly removes the summed dimension $(k)$, leaving only the $i$ dimension (size 2). To explicitly get the column vector we can do `ik,k->i1`

## Matrix-Matrix Multiplication

$$ C_{ij} = ∑_k A_{ik} B_{kj} = A_{ik} B_{kj} $$

```py
a = torch.arange(6).reshape(2, 3)
b = torch.arange(15).reshape(3, 5)
torch.einsum('ik,kj->ij', [a, b])
tensor([[  25.,   28.,   31.,   34.,   37.],
        [  70.,   82.,   94.,  106.,  118.]])
```

## Dot Product

### Vector

$$ c = ∑_i a_{i} b_{i} = a_{i} b_{i} $$

```py
a = torch.arange(3) # -- [0, 1, 2]
b = torch.arange(3,6)  # -- a vector of length 3 containing [3, 4, 5]
torch.einsum('i,i->', [a, b])
tensor(14.)
```

### Matrix

$$ c = ∑_i ∑_j A_{ij} B_{ij} = A_{ij} B_{ij} $$

```py
a = torch.arange(6).reshape(2, 3)
b = torch.arange(6,12).reshape(2, 3)
torch.einsum('ij,ij->', [a, b])
tensor(145.)
```

> Oooo this is elegant!

## Hadamard Product (element-wise product)

$$ C_{ij} = A_{ij} B_{ij} $$

```py
a = torch.arange(6).reshape(2, 3)
b = torch.arange(6,12).reshape(2, 3)
torch.einsum('ij,ij->ij', [a, b])
tensor([[  0.,   7.,  16.],
        [ 27.,  40.,  55.]])
```

## Outer Product

$$ C_{ij} = a_i b_j $$

```py
a = torch.arange(3)
b = torch.arange(3,7)  # -- a vector of length 4 containing [3, 4, 5, 6]
torch.einsum('i,j->ij', [a, b])
tensor([[  0.,   0.,   0.,   0.],
        [  3.,   4.,   5.,   6.],
        [  6.,   8.,  10.,  12.]])
```

> Cleannnnn!

## Batch Matrix Multiplication (Rahul AR 😢)

$$ C_{ijl} = ∑_k A_{ijk} B_{ikl} = A_{ijk} B_{ikl} $$

```py
a = torch.randn(3,2,5)
b = torch.randn(3,5,3)
torch.einsum('ijk,ikl->ijl', [a, b])
tensor([[[ 1.0886,  0.0214,  1.0690],
         [ 2.0626,  3.2655, -0.1465]],

        [[-6.9294,  0.7499,  1.2976],
         [ 4.2226, -4.5774, -4.8947]],

        [[-2.4289, -0.7804,  5.1385],
         [ 0.8003,  2.9425,  1.7338]]])
```

## Tensor Contraction (TODO)

```text
Batch matrix multiplication is a special case of a tensor contraction. Let's say we have two tensors, an order-n
 tensor A∈RI1×⋯×In
 and an order-m
 tensor B∈RJ1×⋯×Im
. As an example, take n=4
, m=5
 and assume that I2=J3
 and I3=J5
. We can multiply the two tensors in these two dimensions (2
 and 3
 for A
 and 3
 and 5
 for B
) resulting in a new tensor C∈RI1×I4×J1×J2×J4
 as follows
Cpstuv=∑q∑rApqrsBtuqvr=ApqrsBtuqvr

a = torch.randn(2,3,5,7)
b = torch.randn(11,13,3,17,5)
torch.einsum('pqrs,tuqvr->pstuv', [a, b]).shape
torch.Size([2, 7, 11, 13, 17])
```

## Bilinear Transformation

As mentioned earlier, **einsum can operate on more than two tensors**. One example where this is used is bilinear transformation.

$$ D_{ij} = ∑_k ∑_l A_{ik} B_{jkl} C_{il} = A_{ik} B_{jkl} C_{il} $$

```py
a = torch.randn(2,3)
b = torch.randn(5,3,7)
c = torch.randn(2,7)
torch.einsum('ik,jkl,il->ij', [a, b, c])
tensor([[ 3.8471,  4.7059, -3.0674, -3.2075, -5.2435],
        [-3.5961, -5.2622, -4.1195,  5.5899,  0.4632]])
```

## 3 Case Studies (TODO)

### Attention

The word-by-word attention mechanism

$$
\begin{align*}

M_t  &= \tanh(W^y Y + (W^h h_t + W^r r_{t−1}) ⊗ e_L)  \quad M_t ∈ ℝ^{k×L} \\
α_t &= \textrm{softmax}(w^T M_t)  \quad α_t ∈ ℝ^{L} \\
r_t &= Y α_{t}^{T} + \tanh(W^t r_{t−1}) \quad r_t ∈ ℝ^{k} \\

\end{align*}
$$

This is not trivial to implement, particularly if we care about a batched implementation. Einsum to the rescue!

```py
# Parameters
# -- [hidden_dimension]
bM, br, w = random_tensors([7], num=3, requires_grad=True)

# -- [hidden_dimension x hidden_dimension]
WY, Wh, Wr, Wt = random_tensors([7, 7], num=4, requires_grad=True)

# Single application of attention mechanism
def attention(Y, ht, rt1):
    # -- [batch_size x hidden_dimension]
    tmp = torch.einsum("ik,kl->il", [ht, Wh]) + torch.einsum("ik,kl->il", [rt1, Wr])
    Mt = F.tanh(torch.einsum("ijk,kl->ijl", [Y, WY]) + tmp.unsqueeze(1).expand_as(Y) + bM)

    # -- [batch_size x sequence_length]
    at = F.softmax(torch.einsum("ijk,k->ij", [Mt, w]))

    # -- [batch_size x hidden_dimension]
    rt = torch.einsum("ijk,ij->ik", [Y, at]) + F.tanh(torch.einsum("ij,jk->ik", [rt1, Wt]) + br)
    
    # -- [batch_size x hidden_dimension], [batch_size x sequence_dimension]
    return rt, at

# Sampled dummy inputs
# -- [batch_size x sequence_length x hidden_dimension]
Y = random_tensors([3, 5, 7])

# -- [batch_size x hidden_dimension]
ht, rt1 = random_tensors([3, 7], num=2)

rt, at = attention(Y, ht, rt1)
at  # -- print attention weights
tensor([[ 0.1150,  0.0971,  0.5670,  0.1149,  0.1060],
        [ 0.0496,  0.0470,  0.3465,  0.1513,  0.4057],
        [ 0.0483,  0.5700,  0.0524,  0.2481,  0.0813]])
```

### TreeQN (read through again)

An example where I (@timvieira) used einsum in the past is implementing equation 6 in 8. Given a low-dimensional state representation $z_l$ at layer $l$ and a transition function $W^a$ per action $a$, we want to calculate all next-state representations $z_{l+1}^{a}$ using a residual connection.

$$ z_{l+1}^{a} = z_l + \tanh(W^a z_l) $$

In practice, we want to do this efficiently for a batch $B$ of $K$-dimensional state representations $Z∈ℝ^{B×K}$ and for all transition functions (i.e. for all actions $A$) at the same time. We can arrange these transition functions in a tensor $W∈ℝ^{A×K×K}$ and calculate the next-state representations efficiently using einsum.

```py
import torch.nn.functional as F

def random_tensors(shape, num=1, requires_grad=False):
  tensors = [torch.randn(shape, requires_grad=requires_grad) for i in range(0, num)]
  return tensors[0] if num == 1 else tensors

# Parameters

# -- [num_actions x hidden_dimension]
b = random_tensors([5, 3], requires_grad=True)

# -- [num_actions x hidden_dimension x hidden_dimension]
W = random_tensors([5, 3, 3], requires_grad=True)

def transition(zl):
  # -- [batch_size x num_actions x hidden_dimension]
  return zl.unsqueeze(1) + F.tanh(torch.einsum("bk,aki->bai", [zl, W]) + b)

# Sampled dummy inputs
# -- [batch_size x hidden_dimension]
zl = random_tensors([2, 3])

transition(zl)
tensor([[[ 0.9986,  1.9339,  1.4650],
         [-0.6965,  0.2384, -0.3514],
         [-0.8682,  1.8449,  1.5787],
         [-0.8050,  0.7277,  0.1155],
         [ 1.0204,  1.7439, -0.1679]],

        [[ 0.2334,  0.6767,  0.5646],
         [-0.1398,  0.7524, -0.9820],
         [-0.8377,  0.4516, -0.3306],
         [ 0.4742,  1.1055,  0.1824],
         [ 0.8868,  0.2930,  0.1579]]])

```

## Summary

Einsum is one function to rule them all. It's your Swiss Army knife for all kinds of tensor operations. That said, "einsum is all you need" is obviously an overstatement. If you look at the case studies above, we still need to apply non-linearities and construct extra dimensions (unsqueeze). Similarly, for splitting, concatenating or indexing of tensors you still have to employ other library functions. 

> NOTE: Moreover, einsum in PyTorch currently does not support diagonal elements, so the following throws an error: `torch.einsum('ii->i', [torch.randn(3, 3)])`.

One thing that can become annoying when working with einsum is that you have to instantiate parameters manually, take care of their initialization, and registering them with modules. Still, I strongly encourage you to look out for situations where you can use einsum in your models.
