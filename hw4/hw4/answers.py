r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.5,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # DONE: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=5,
              gamma=0.99,
              beta=0.2,
              learn_rate=0.008,
              eps=1e-8,
              hidden_dims=[5, 5],
              nonlin='relu',
              dropout=0
              )
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=1.,
              delta=1.,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # DONE: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=5,
              gamma=0.99,
              beta=0.2,
              delta=1.,
              learn_rate=0.008,
              eps=1e-8,
              actor=dict(hidden_dims=[16, 16, 16],
                         nonlin='relu',
                         dropout=0),
              critic=dict(hidden_dims=[16, 16],
                          nonlin='relu',
                          dropout=0.5)
              )
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q2 = r"""
**Your answer:**

To understand why we get a valid approximation when using the estimated q-values as regression targets for the state 
values we should recall the definitions of both functions:

**$v_\pi(s)$** expresses the expected value of the discounted reward that would be gained by following the policy 
$\pi$ starting from state s:

$$
\begin{align}
v_{\pi}(s) &= \E{g(\tau)|s_0 = s,\pi} \\
\end{align}
$$

In contrast, **$q_{\pi}(s,a)$** expresses the expected value of the discounted reward that would be gained by following
the policy  $\pi$ starting from state s **after fixing the first action - a**, which does **not** necessarily depend on
$\pi$:

$$
\begin{align}
q_{\pi}(s,a) &= \E{g(\tau)|s_0 = s,a_0=a,\pi}.
\end{align}
$$

From here we can see that $v_\pi(s)$ can be expressed by $q_{\pi}(s,a)$ like so:

$$
\begin{align}
v_{\pi}(s) &= \E{q_{\pi}(s,a)|s_0 = s, a\in A, \pi} \\
\end{align}
$$

Where $A$ is the group of all possible actions.


Finally, we should also note that in the case of AAC critic learning the first actions which define the estimated
q-values (which are used as the regression targets) **were all selected by the policy $\pi$**. This means that the 
probability for these actions in each respective state was the greatest compared to the other possible actions. Thus,
by definition of expectation we get that if $a$ is the most probable action according to $\pi$ in state $s$, then the 
expected reward gained by the trajectories starting with $a$ will have the most effect on the value of $v_{\pi}(s)$.
Therefore, $q_{\pi}(s,a)$, which takes into account **only** the trajectories starting with $a$ from state $s$, serves 
as a pretty good approximation for the state value of  $s$. We can therefore conclude that using the estimated q-values
as regression targets for the state values would probably lead to a valid approximation.

"""


part1_q3 = r"""
**Your answer:**

1. Here are our conclusions from the first experiment:


First, in the mean_reward graph it is evident that as expected the Vanilla
PG (vpg) has achieved the worst results. This is not surprising because this PG suffers from high variance and from narrow
policy distribution which limits the agent's tendency to explore new directions. Next, it can be seen that the Entropy 
PG (epg) has managed to get better results compared to vpg, which can be attributed to the fact the this gradient
tries to maximize the entropy of the policy distribution, thus enforcing the agent to explore new (and maybe better)
directions. Lastly, it appears that the best policy gradients were the Combined (cpg) and the Baseline (bpg) policy 
gradients. This is not surprising because these gradients, unlike the other two, use a baseline to reduce their variance,
which leads to a more stable optimization behavior and thus to faster convergence.

Another interesting observation is that policy gradients which try to maximize the entropy seem to converge more slowly 
compared to the gradients which don't (for example: cpg converges more slowly compared to bpg). This is probably 
because maximize the entropy forces the agent to explore more directions, which is an advantage but can also slow down
convergence in the right directions.

"""
