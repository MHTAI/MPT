# Modern Portfolio Theory

This is the implementation of the Modern Portfolio Theory. The objective of this project is to find the optimized Sharpe Ratio through:

- SciPy Optimization: Utilizes the scipy.optimize module to  calculate the optimal Sharpe Ratio;
- Monte Carlo Simulation: Involves generating a large number of random samples to estimate and identify the best portfolio configuration.

Afterwards, we will plot the Efficient Frontier with the data above for better visualization.

![download](https://github.com/user-attachments/assets/25922506-02f4-4459-882c-7025f3924c60)

# Explanation

Considering a portfolio composed of a range of assets (Asset A, Asset B, ..., Asset Z), each with its own volatility and weighting, the portfolio return should be:
<br/>

$$
E(R_p) = \sum_{i=a}^{z} w_i E(R_i)
$$

<br/><br/>
while the portfolio variance should be:

$$
\sigma_P^2 =
\begin{bmatrix}
    w_a & w_b & w_c & \cdots & w_z
\end{bmatrix}
\times
\begin{bmatrix}
    \sigma_a^2 & \text{Cov}(a, b) & \text{Cov}(a, c) & \cdots & \text{Cov}(a, z)\\
    \text{Cov}(a, b) & \sigma_b^2 & \text{Cov}(b, c) & \cdots & \text{Cov}(b, z)\\
    \text{Cov}(a, c) & \text{Cov}(b, c) & \sigma_c^2 & \cdots & \text{Cov}(c, z)\\
    \vdots & \vdots & \vdots & \ddots & \vdots\\
    \text{Cov}(a, z) & \text{Cov}(b, z) & \text{Cov}(c, z) & \cdots & \sigma_z^2
\end{bmatrix}
\times
\begin{bmatrix}
    w_a & w_b & w_c &\cdots & w_z
\end{bmatrix}^{T}
$$

<br/><br/>
In short:

$$
\sigma_P^2 =
W \cdot M \cdot W^T
$$

Hence:

$$
\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}
$$

We could calculate the Sharpe Ratio through adjusting the weights but we set the return value as negative Sharpe Ratio. The reason is SciPy only provides the minimize function.
<br/><br/>

![image](https://github.com/user-attachments/assets/61f03b76-4445-4bac-b9dc-b1427140d549)

We use the function to find the lowest value of the negative Shape Ratio. Minimizing the negative Sharpe ratio is equivalent to maximizing the Sharpe ratio.
