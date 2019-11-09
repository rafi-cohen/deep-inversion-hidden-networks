r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


It appears that the best accuracy scores were achieved using relatively low values of k (1-5), while the best score was
achieved using `k = 3`.

On the one hand, by basing its prediction on a very small subset of the sample's closest neighbors the model might tend
to overfit to the training data, which would lead to worse generalization for unseen data. On the other hand, basing
the model's prediction on a very large subset of the neighbors might introduce unwanted noise into the classification 
process and ultimately lower the model's accuracy. This would happen because in this case unrelated neighbors that are
farther away from the sample might have too much impact on the model's prediction.
"""

part2_q2 = r"""
**Your answer:**


1. Selecting the best model with respect to the train-set accuracy basically means choosing the most overfitted model.
    Doing so is against the idea of selecting a model that can make generalized predictions on unseen data.

    On the other hand, while using k-fold CV, we are able to test the model's accuracy on unseen data
    even *during* the training process, which allows us to asses its generalization ability for future unseen datasets.

2. Selecting the best model with respect to the test-set accuracy after training the model on the entire
    train-set might hinder the generalization of the model on unseen data. The reason for that is that we don't 
    know whether the test-set is a good representation of the real distribution of the data, and therefore selecting
    our parameters based on the accuracy on this set could lead to biased results.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
