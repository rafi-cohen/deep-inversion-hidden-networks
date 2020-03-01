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

This question has already been answered in the notebook (we sent an email to Aviv and he said it is ok to leave it as it
is):

"""


part1_q2 = r"""
**Your answer:**

To understand why we get a valid approximation when using the estimated q-values as regression targets for the state 
values we should recall the definitions of both functions:

**$v_\pi(s)$** expresses the expected value of the discounted reward that would be gained by following the policy 
$\pi$ starting from state $s$:

$$
\begin{align}
v_{\pi}(s) &= \E{g(\tau)|s_0 = s,\pi} \\
\end{align}
$$

In contrast, **$q_{\pi}(s,a)$** expresses the expected value of the discounted reward that would be gained by following
the policy  $\pi$ starting from state $s$ **after fixing the first action - $a$**, which does **not** necessarily depend on
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


Finally, we should also note that in the case of the AAC critic learning the first actions which define the estimated
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
    gradients. This is not surprising because these gradients, unlike the other two, use a baseline to reduce variance,
    which leads to a more stable optimization behavior and thus to faster convergence.
    
    Another interesting observation is that policy gradients which try to maximize the entropy seem to converge more slowly 
    compared to the gradients which don't (for example: cpg converges more slowly compared to bpg). This is probably 
    because maximizing the entropy forces the agent to explore more directions, which is an advantage but can also slow down
    convergence in the right directions.
    
    In the baseline graph it can be seen, as expected, that both curves are increasing. This makes sense because the baseline
    we used in this experiment was the average of the estimated state-values $\hat{q}_{i,t}$, and as the policy get better
    gradually during the learning procedure this average naturally gets higher.
    
    In the loss_p graph we can see as expected that all loss curves get smaller (in absolute value) as the learning 
    procedure advances. Two interesting observations are that the loss curves of the bpg and cpg are very close to zero
    during the entire training process and are looking significantly more stable. This makes sense since these two policy
    gradients use a baseline which leads to a significant reduction of their variance.
    
    In the loss_e graph we also see that all loss curves get smaller (in absolute value) as the learning 
    procedure advances. This makes sense since as the learning process advances and the policy improves, the policy 
    distribution gets narrower (since the best actions gets higher probabilities) and thus the entropy gets smaller.

2. The graphs clearly show that the aac PG is vastly superior to the rest of the policy gradients, and in particular to
the cpg. In the mean_reward graph we can see that while cpg achieved its maximum reward value of 121.5 in about 3600
episodes, aac was able to pass that mark in only about 1200 episodes, which illustrates how much faster it converges.
Furthermore, in 4000 episodes aac was able to achieve a maximum reward of 241 (!), which is about double the maximum reward
of cpg. These gains can be attributed to aac's improved baseline, which reduces its variance even further compared
to cpg, thus creating a more stable optimization behaviour which allows faster convergence.

"""
