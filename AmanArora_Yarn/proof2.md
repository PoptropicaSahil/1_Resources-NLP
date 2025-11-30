# Proof about Rotation Matrix

This rotation matrix comes from basic trigonometry. When rotating a point
$(x, y)$ by angle $\theta$
 counter-clockwise around the origin:

1. **Starting with polar coordinates:** Any point $(x, y)$ can be written as $(r\cos\alpha, r\sin\alpha)$

2. After rotation: The new angle becomes
$\alpha + \theta$, giving us the new point $(r\cos(\alpha + \theta), r\sin(\alpha + \theta))$.

3. Using trigonometric identities:

```math

\begin{align*}
    x' &= r\cos(\alpha + \theta) &= r(\cos\alpha\cos\theta - \sin\alpha\sin\theta) &= x\cos\theta - y\sin\theta \\
    y' &= r\sin(\alpha + \theta) &= r(\sin\alpha\cos\theta + \cos\alpha\sin\theta) &= x\sin\theta + y\cos\theta
\end{align*}
```

4. Matrix form: This gives us:

```math

\begin{pmatrix} 
    x' \\ y' 
\end{pmatrix} 
= 
\begin{pmatrix} 
    \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta 
\end{pmatrix} 
\begin{pmatrix}
    x \\ y 
\end{pmatrix}
```
