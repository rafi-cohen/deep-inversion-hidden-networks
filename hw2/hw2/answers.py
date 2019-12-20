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
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. Our expectation was that the models using dropout would be less overfitted than the one without, which 
would translate to better generalization (better accuracy) on the test set. 

    The graphs did in fact match our expectation: While the model
without the dropout reached almost 90% accuracy on the training set, it performed a
lot worse on the test set than the models with the dropout, even though these models did not perform as well
on the training set. This proves that the model without the dropout was extremely overfitted, which indicates
that using dropout does prevent overfitting and thus leads to better generalization.

2. The graphs indicate that during the training phase the low-dropout setting performed significantly better
than the high-dropout setting (~65% vs. ~30%). This, along with the test accuracy results (~28% vs. ~25%) perhaps
indicate that the low-dropout model is more overfitted than the high-dropout one, but since the lower model achieved
better test accuracy while also converging faster, we think it makes a pretty good compromise between not using dropout
at all, and using dropout at a probability that is too high. We expect however that the ideal dropout value in this case
would be somewhere in between the 2 values we used.

"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible. Example:
Assuming we have 2 samples `x1`,`x2` with ground-truth labels: `Y1`,`Y2` respectively.
It is possible that in epoch `t` the probability vectors (in the form $[Y1_{score}, Y2_{score}]$) we would get for each
sample are: $[0.4, 0.6]$ for `x1` and $[0, 1]$ for `x2`. Therefore both samples would be classified as belonging to `Y2`
which means: accuracy = 50%, while the CE loss is: $\text{loss}=-\log(0.4)-\log(1)=0.916$.

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
the network was unable to learn at all using larger values of `L`. We suspect that the reason for that
is the exploding/vanishing gradient phenomenon which we discussed in class. After looking into the matter,
we discovered that we were in fact dealing here with vanishing gradients.

2. The values of `L` with which the network was unable to learn are `L=16` for both values of `K`,
and `L=8` for `K=32`. As stated above, the reason for that is the vanishing gradient phenomenon, which
gets worse in deeper networks. This happens because in deeper networks the gradients are multiplied more times
than in shallow networks (because of the extra layers). Thus, if the magnitudes of the values in these
multiplications are too small (between 0 and 1), in the deeper networks the product of many small values
might get too close to 0 which would prevent these networks from learning properly. 

    One way to try to resolve this issue is using batch-normalization, which helps stabilizing the gradients
as we discussed in class.

    Another way is to use residual networks. These networks use skip connections, which might help
mitigate the problem by allowing the gradients to flow backwards more freely through the skip
connections and thus making them less prone to vanishing.
"""

part3_q2 = r"""
**Your answer:**

By looking at the graphs, it appears that large values of `K` (large number of filters) don't work well
in shallow networks while smaller values work fine. For example, in the case of `L=2` and `K=256`, the network
achieved significantly worse results, compared to configurations with other values of `K` with the same `L`.
On the other hand, it appears that in deeper networks, smaller values of `K` don't work so well while larger
values work fine. For example, for `L=4` and `L=8` the networks using the smallest possible value of
`K` (32) achieved the worst results.
 
These two conclusions might hint of a positive linear connection between the number of filters used
in the networks and how deep it should be in order to get good results.
 
It appears, in both experiments 1.1 and 1.2, that in some cases deeper networks were not able to learn. Again, this is a
direct result of the vanishing gradients phenomenon which we explained in our answer to Q1.

"""

part3_q3 = r"""
**Your answer:**

Not surprisingly, the best results in experiment 1.3 were achieved by the shallowest networks (`L=1,2`) with
the three/six convolutional layers. Not only did these networks achieve the best accuracy scores, they also had the
fastest convergence rate (`L=1` achieved over 70% accuracy in about 5 epochs, `L=2` in about 9 epochs, while it took
`L=3` over 20 epochs to reach the same accuracy). So overall it seems that the deeper the network, the slower its
convergence rate is. Also, once again, it seems that once the network gets too deep (`L=4`, with 12 layers) it is unable 
to learn due to the vanishing gradient phenomenon.

"""

part3_q4 = r"""
**Your answer:**

Once again, it seems that the shallowest networks achieved the best results, but this time by a significantly
wider margin. For example, for `K=32`, the network using `L=8` achieved about 12% better test accuracy than the
networks with the largest `L` values. While in the case of `K=[64, 128, 256]`, the network with `L=2` achieved
the best results and also converged the fastest. It should also be noted that for the first time, results close
to 80% test accuracy were seen, which illustrates the advantage of ResNets over regular CNNs.

Also, the results of this experiment justify our previous suggestion for using ResNets as a way to deal
with the vanishing gradient phenomenon. For example, while in experiment 1.1, the (`L=8`, `K=32`) network was
completely unable to learn because of the above phenomenon, the same architecture using ResNet performed very well,
and was able to achieve 70% test accuracy.

Furthermore, compared to experiment 1.3, where the 12-layer (`L=4`, `K=[64,128,256]`) network was unable to learn,
the same architecture using ResNet again performed very well, and even the 24-layer (`L=8`, `K=[64,128,256]`) network
was able to converge. This emphasizes how capable the skip connections are in preventing vanishing gradients.

"""

part3_q5 = r"""
**Your answer:**
1. Our main modifications to the architecture were the following:

    ##### <u>BatchNorm layer after each activation function:</u>
    Since we want our architecture to support deeper networks, we concluded that we need some way to prevent the
    vanishing gradient phenomenon. As we explained in Q1, one way to try to resolve this issue is by using batch 
    normalization, which helps stabilizing the gradients.
    
    ##### <u>Dropout layer after every MaxPool layer:</u>
    As we learned in class (and empirically witnessed in part 2), dropout layers can be used to help improve the 
    generalization of networks by reducing overfitting. Thus, we hoped adding this feature to our architecture 
    would help improve its accuracy.
    
    ##### <u>No hidden layers in the classifier:</u>
    After empirically testing several different architectures with different configurations and comparing their results, 
    we eventually decided not to use any hidden layers in the classifier. 
    
        
2. We tested our architecture on all configurations using: `K=[32,64,128]` and `L=[2,3,6,9,12]`. Since most of these 
configurations create pretty deep networks (for example using `L=12` leads to 36 convolutional layers), we weren't
sure whether some of these networks would be able to learn at all given that significantly shorter networks weren't able
to in the previous experiments. To our surprise, all the networks we tested using this architecture, no matter how deep,
were able to learn this time, which proves how capable batch normalization is in preventing vanishing gradients. 
Also, in its best configuration (`L=3`) our architecture peaked at 84.9% test accuracy, which is significantly better 
than even the best ResNet configuration we tested in experiment 1.4, which barely hit 80%. We think this improvement in
accuracy can be attributed to the fact that our architecture, unlike previous architectures, utilizes dropout layers 
which improves its generalization.
"""
# ==============
