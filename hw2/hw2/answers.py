r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # DONE: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.025
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # DONE: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.025
    lr_momentum = 0.005
    lr_rmsprop = 0.00025
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # DONE: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.0025
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. Our expectation was that the models using dropout would be less overfitted than the one without, which 
would translate to better generalization (better accuracy) on the test set. 

    The graphs did in fact match our expectation: While the model
without the dropout reached almost 100% accuracy on the training set, it performed a
lot worse on the test set than the models with the dropout, even though these models did not perform as well
on the training set. This proves that the model without the dropout was extremely overfitted, which indicates
that using dropout does prevent overfitting and thus leads to better generalization.

2. The graphs indicate that during the training phase the low-dropout setting performed significantly better
than the high-dropout setting. This might lead to think that the low-dropout setting is more overfitted than 
the high-dropout one, but the graph of the test accuracy indicates otherwise: It seems that the low-dropout model
was able to converge faster than the high-dropout model, which shows that the lower value is a good compromise
between not using dropout at all, and using dropout at a probability that is too high. We expect however that the best
dropout value would be somewhere in between the 2 values we used.

"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible. Example:
Assuming we have 2 samples `x1`,`x2` with ground-truth labels: `Y1`,`Y2` respectively.
It is possible that in epoch `t` the probability vectors (in the form $[Y1_{score}, Y2_{score}]$) we get for each sample
are: $[0.4, 0.6]$ for `x1` and $[0, 1]$ for `x2`. Therefore both samples would be classified as belonging to `Y2` which
means: accuracy = 50%, while the CE loss is: $\text{loss}=-\log(0.4)-\log(1)=0.916$.

Later, it is also possible that in epoch `t+1` the probability vectors we get for each sample
are: $[0.6, 0.4]$ for `x1` and $[0.4, 0.6]$ for `x2`. Therefore both samples would be classified correctly which
means: accuracy = 100%, while the CE loss is: $\text{loss}=-\log(0.6)-\log(0.6)=1.021$.

Thus we have shown that while the accuracy increased from 50% to 100%, the loss also increased from 0.916 to 1.021.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

1. It appears that the network achieved better accuracy using smaller values of `L` (depth).
The value of `L` which was used to achieve the best results was 2. Also, it appears that in some cases
the network was unable to learn at all using larger values of `L`. We suspected that the reason for that
was the exploding/vanishing gradient phenomenon which we discussed in class. After looking into the matter,
we discovered that we are dealing here with vanishing gradient.

2. The values of `L` with witch the network was unable to learn are `L`=16 for both values of `K`,
and `L`=8 for `K`=32. As stated above, the reason for that is the vanishing gradient phenomenon, which
gets worse in deeper networks. This happens because in these networks the gradients are multiplied more times
than in shallow networks (because of the extra layers). Thus, if the magnitudes of the values in the
multiplications are too small (between 0 and 1), in the deeper networks the product of many small values
might get too close to 0 which would prevent these networks from learning properly. 

    One way to resolve this issue is using batch-normalization, which helps stabilizing the gradients
as we discussed in class.

    Another way is to use residual networks. These networks use skip connections, which might help
mitigate the problems because the gradients are able to flow backwards more freely through the skip
connections and are thus less prone to vanishing.
"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
