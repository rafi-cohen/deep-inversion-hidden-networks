r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


It appears that the best accuracy scores were achieved using relatively low values of `k` (1-5), while the best score was
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
    
    However, in k-fold CV, we are evaluating our model on multiple different independent validation sets, and
    then selecting the model with the best average results. This reduces the risk of the model being biased
    towards a single set.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


The selection of $\Delta > 0$ is arbitrary for the SVM loss $L(\mat{W})$ because the loss, as defined above,
is affected by both the value of $\Delta$ and the distances between the scores.
However, the hyperparameter $\lambda$ controls the magnitude of the weights, therefore allowing us
to stretch/shrink them and by doing so we can effectively control the distances between the scores.
Therefore given a model with the hyperparameters $\Delta_1, \lambda_1$, and another model
with the hyperparameter $\Delta_2\neq\Delta_1$, we can find a $\lambda_2$ such that both models will behave the same.
This means that we can safely set $\Delta$ at a fixed value (e.g. $\Delta=1$) and then just tune the $\lambda$
hyperparameter, instead of tuning both parameters. 

"""

part3_q2 = r"""
**Your answer:**

It is clear that the brighter parts of most of the weights images resemble the digits that these weights represent.
This can be explained given how the model calculates the scores for the images:

$$
s_j = \vectr{w_j} x + b_j.
$$

Given an image vector $\vectr{x}$, the model calculates the compatibility score of $\vectr{x}$ for each digit class j by multiplying 
$\vectr{x}$ by $\vectr{w_j}$ (and adding the bias). We should note that the values of $\vectr{x}$ are in the range [0,255] in order to represent
different shades of gray (0 for white, 255 for black). Pixels in $\vectr{x}$ that create the actual shape of the digit are in white
and therefore have lower values. That's why in order to maximize the score of $\vectr{w_j} x$ (where $\vectr{w_j}$ is the
weight of the correct class of $\vectr{x}$), in the indices of the pixels where $\vectr{x}$ has the black background $\vectr{w_j}$ must have values
that will 'cancel them out', whereas in the indices in $\vectr{x}$ of the actual digit $\vectr{w_j}$ must have values
that will magnify them. That's why on average the image of each weight will look similar to the digit of the class it is
representing after the training process.

This can explain some of the classification errors seen above, for example: the digit 5 was mistakenly classified as
the digit 6 since it appears they have relatively similar weight vectors (the images of their weights look similar).

The way this model's interpretation is similar to KNN is in the following regard:
* Whereas KNN classifies each sample by looking at its nearest neighbor in the training set, SVM models the common
traits of each class digit in its weights vector ($\vec{w_j}$) and then finds the "closest" class to the sample
($\vec{x_i}$) by choosing the class whose score ($\vec{w_j} \vec{x_i}$) is the greatest. Therefore, the classes can be seen as
the neighbors, the scores as the distances (where a greater score means a smaller distance),
and SVM chooses the "nearest" class.

"""

part3_q3 = r"""
**Your answer:**


1. The curve of the training set loss in our graph is pretty smooth, and therefore we think that value we chose for the
learning rate ($\eta=0.025$) is good. Had we chosen a value which is too low, the algorithm would not have been able
to converge enough in 30 epochs, and therefore the loss wouldn't have gone as low as it has in our graph.
On the other hand, had we chosen a value which is too high, the algorithm would've converged faster, but after
getting close to the local minimum, the higher learning rate would've made it sway back and forth close to that point,
which would've translated into spikes in the loss curve.

2. Based on the difference between the accuracy curves on the train and validation sets, we think that our model is
slightly overfitted to the training set. It appears that after about 10 epochs the accuracy of the model on the
validation set has reached its peak (about 90%), and after that it did not seem to improve any longer. On the other
hand, the accuracy of the model on the training set *did* continue to improve even after the 10th epoch, which indicates
that the model started overfitting itself to the training data in the later epochs.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


The ideal pattern to see in a residual plot is a pattern where all of the points are on a straight horizontal line that 
goes through 0 in the vertical axis. This means that the magnitude of the residual $e^{(i)} = y^{(i)} - \hat{y}^{(i)}$
is zero for all samples, meaning that all predictions are identical to their true value.
However, in most cases this level of accuracy is unattainable but we would still want all the points to be inside a narrow
horizontal strip around the value of 0. The width of the strip (marked by the dotted lines above) is determined by the
std value of the values of the points: the smaller the std, the more concentrated the points will be (hopefully) 
around the zero line.

By looking at the two residual plot graphs that we have created, it is clear the pattern we described above is much
more evident in the CV plot: in this plot almost all the points are concentrated inside a narrow strip around the zero line,
whereas in the top-5 plot the points are significantly more scattered around a slightly wider strip. These results are not surprising as 
the model corresponding the the final plot is significantly more sophisticated than the model of the top-5 plot (a large set of non-linear featuers
vs. a very small set of linear features).

"""

part4_q2 = r"""
**Your answer:**

1. We think `np.logspace` was used to define the range for $\lambda$ because using a logarithmic scale allows us to easily
generate a relatively small set of values that differ from each other in multiple orders of magnitude. This property is desired
because it allows us to quickly find the order of magnitude of the best value of $\lambda$ without having to scan a huge set
of values. Then, after finding the optimal order of magnitude, one can fine-tune the results even further by looking
for the best value of $\lambda$ of that magnitude. On the other hand, this task is not possible using
`np.linspace` which generates a set of numbers that are evenly spaced.

2. The number of times the model was fitted to the data can be given by:

$$
|\text{degree_range}| \times |\text{lambda_range}| \times |\text{k_folds}| = 3 \times 20 \times 3 = 180
$$

"""

# ==============
