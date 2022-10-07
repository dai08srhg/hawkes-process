# hawkes-process

## Hawkes Process
The probability density function of hawkes process is given by

$$ p_{[0,T]}(\boldsymbol{t}_n) = \prod_{i=1}^n \left[\mu + \sum_{j < i}g(t-t_j)\right] \times \exp\left[-\mu T -  \sum_{i=1}^n\int_{t_i}^T g(s-t_i)ds \right] $$

where, $g(\tau)$ is a kernel function representing the influence from past events.

### Maximum Likelihood Estimation of a hawkes process
Assume the following for kernel functions
$$g(\tau) = ab \exp(-b\tau)$$

Maximum likelihood estimation of $\boldsymbol{\theta} = \{\mu, a, b\}$

The likelihood function $L(\boldsymbol{\theta}|\boldsymbol{t}_n)$ is 
$$
L(\boldsymbol{\theta}|\boldsymbol{t}_n) = p_{[0,T]}(\boldsymbol{t}_n | \boldsymbol{\theta})
$$
$$\hat{\boldsymbol{\theta}} = \argmin_{\boldsymbol{\theta}}-\log L(\boldsymbol{\theta}|\boldsymbol{t}_n) $$

Estimate parameters by gradient descent method.

<div align="center">
<img src="img/intensity.png" width="70%">
<img src="img/expected_occurrences.png" width="50%">
</div>
## Usage



