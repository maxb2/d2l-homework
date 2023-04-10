# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: docutools, -ExecuteTime, -execute, -execution
#     notebook_metadata_filter: docutools, tags, -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: d2l-homework
#     language: python
#     name: d2l-homework
# ---

# %% [markdown]
# # Text

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import random
import torch
from torch.distributions.multinomial import Multinomial

# %%
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(100)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])


# %%
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()

# %%
Multinomial(100, fair_probs).sample() / 100

# %%
counts = Multinomial(10000, fair_probs).sample()
counts / 10000

# %%
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

fig, ax = plt.subplots(figsize = (4.5, 3.5))
ax.plot(estimates[:, 0], label=("P(coin=heads)"))
ax.plot(estimates[:, 1], label=("P(coin=tails)"))
ax.axhline(y=0.5, color='black', linestyle='dashed')
ax.set_xlabel('Samples')
ax.set_ylabel('Estimated probability')
plt.legend();

# %% [markdown]
# # Exercises

# %% [markdown]
# 1. Give an example where observing more data can reduce the amount of uncertainty about the outcome to an arbitrarily low level.
# 1. Give an example where observing more data will only reduce the amount of uncertainty up to a point and then no further. Explain why this is the case and where you expect this point to occur.
# 1. We empirically demonstrated convergence to the mean for the toss of a coin. Calculate the variance of the estimate of the probability that we see a head after drawing $n$ samples.
#     1. How does the variance scale with the number of observations?
#     1. Use Chebyshev's inequality to bound the deviation from the expectation.
#     1. How does it relate to the central limit theorem?
# 1. Assume that we draw $n$ samples $x_i$ from a probability distribution with zero mean and unit variance. Compute the averages $z_m \stackrel{\mathrm{def}}{=} m^{-1} \sum_{i=1}^m x_i$. Can we apply Chebyshev's inequality for every $z_m$ independently? Why not?
# 1. Given two events with probability $P(\mathcal{A})$ and $P(\mathcal{B})$, compute upper and lower bounds on $P(\mathcal{A} \cup \mathcal{B})$ and $P(\mathcal{A} \cap \mathcal{B})$. Hint: graph the situation using a [Venn diagram](https://en.wikipedia.org/wiki/Venn_diagram).
# 1. Assume that we have a sequence of random variables, say $A$, $B$, and $C$, where $B$ only depends on $A$, and $C$ only depends on $B$, can you simplify the joint probability $P(A, B, C)$? Hint: this is a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain).
# 1. In :numref:`subsec_probability_hiv_app`, assume that the outcomes of the two tests are not independent. In particular assume that either test on its own has a false positive rate of 10% and a false negative rate of 1%. That is, assume that $P(D =1 \mid H=0) = 0.1$ and that $P(D = 0 \mid H=1) = 0.01$. Moreover, assume that for $H = 1$ (infected) the test outcomes are conditionally independent, i.e., that $P(D_1, D_2 \mid H=1) = P(D_1 \mid H=1) P(D_2 \mid H=1)$ but that for healthy patients the outcomes are coupled via $P(D_1 = D_2 = 1 \mid H=0) = 0.02$.
#     1. Work out the joint probability table for $D_1$ and $D_2$, given $H=0$ based on the information you have so far.
#     1. Derive the probability of the patient being positive ($H=1$) after one test returns positive. You can assume the same baseline probability $P(H=1) = 0.0015$ as before.
#     1. Derive the probability of the patient being positive ($H=1$) after both tests return positive.
# 1. Assume that you are an asset manager for an investment bank and you have a choice of stocks $s_i$ to invest in. Your portfolio needs to add up to $1$ with weights $\alpha_i$ for each stock. The stocks have an average return $\boldsymbol{\mu} = E_{\mathbf{s} \sim P}[\mathbf{s}]$ and covariance $\boldsymbol{\Sigma} = \mathrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$.
#     1. Compute the expected return for a given portfolio $\boldsymbol{\alpha}$.
#     1. If you wanted to maximize the return of the portfolio, how should you choose your investment?
#     1. Compute the *variance* of the portfolio.
#     1. Formulate an optimization problem of maximizing the return while keeping the variance constrained to an upper bound. This is the Nobel-Prize winning [Markovitz portfolio](https://en.wikipedia.org/wiki/Markowitz_model) :cite:`Mangram.2013`. To solve it you will need a quadratic programming solver, something way beyond the scope of this book.
#

# %%
