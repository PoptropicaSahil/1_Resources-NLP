# Proof about why any other position block at p+k can be got from block at p

For each frequency, the 2-D block $[\sin(\cdot),\cos(\cdot)]$ at position $pos+k$ is obtained from the block at position $pos$ by multiplying with a fixed $2\times 2$ matrix that depends only on $k$, not on $pos$

## Setup

Fix a model dimension $d_{\text{model}}$.
For each even index $2i$ define the frequency

```math
\omega_i = 10000^{-2i/d_{\text{model}}}.
```

Then the positional encoding uses, for that pair of dimensions:

```math
PE_{(pos,2i)} = \sin(\omega_i\, pos),\quad
PE_{(pos,2i+1)} = \cos(\omega_i\, pos).
```

Write this 2-vector as:

```math
v_{pos}^{(i)} =
\begin{bmatrix}
\sin(\omega_i\, pos)\\[2pt]
\cos(\omega_i\, pos)
\end{bmatrix}.
```

The full $PE_{pos}$ is just the concatenation of these 2-vectors over all $i$.

## Linear relation for a fixed offset

Consider position $pos+k$. For the same frequency $\omega_i$:

```math
v_{pos+k}^{(i)} =
\begin{bmatrix}
\sin(\omega_i(pos+k))\\[2pt]
\cos(\omega_i(pos+k))
\end{bmatrix}.
```

Use the angle–addition formulas:

```math
\sin(a+b) = \sin a\cos b + \cos a\sin b,\quad
\cos(a+b) = \cos a\cos b - \sin a\sin b.
```

Set $a=\omega_i pos$, $b=\omega_i k$. Then:

```math
\sin(\omega_i(pos+k)) 
= \sin(\omega_i pos)\cos(\omega_i k)
 + \cos(\omega_i pos)\sin(\omega_i k),
```

```math
\cos(\omega_i(pos+k))
= \cos(\omega_i pos)\cos(\omega_i k)
 - \sin(\omega_i pos)\sin(\omega_i k).
```

In matrix form:

```math
\begin{bmatrix}
\sin(\omega_i(pos+k))\\[2pt]
\cos(\omega_i(pos+k))
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
\cos(\omega_i k) & \sin(\omega_i k)\\[2pt]
-\sin(\omega_i k) & \cos(\omega_i k)
\end{bmatrix}
}_{R_i(k)}
\begin{bmatrix}
\sin(\omega_i pos)\\[2pt]
\cos(\omega_i pos)
\end{bmatrix}.
```

So for each frequency $i$:

```math
v_{pos+k}^{(i)} = R_i(k)\, v_{pos}^{(i)}.
```

where $R_i(k)$ depends only on the offset $k$.

## Assembling the full encoding

The full positional encoding vector is:

```math
PE_{pos} = 
\begin{bmatrix}
v_{pos}^{(0)}\\
v_{pos}^{(1)}\\
\vdots
\end{bmatrix},\quad
PE_{pos+k} =
\begin{bmatrix}
v_{pos+k}^{(0)}\\
v_{pos+k}^{(1)}\\
\vdots
\end{bmatrix}.
```

Using the per-frequency relation, this becomes:

```math
PE_{pos+k} =
\begin{bmatrix}
R_0(k) & & \\
& R_1(k) & \\
& & \ddots
\end{bmatrix}
\begin{bmatrix}
v_{pos}^{(0)}\\
v_{pos}^{(1)}\\
\vdots
\end{bmatrix}
= R(k)\, PE_{pos},
```

where $R(k)$ is block-diagonal with the $R_i(k)$ blocks.

Thus, for any fixed offset $k$, there exists a fixed matrix $R(k)$ such that:

```math
PE_{pos+k} = R(k)\, PE_{pos}
```

for all $pos$, meaning the encoding depends only on the relative offset $k$.

---
