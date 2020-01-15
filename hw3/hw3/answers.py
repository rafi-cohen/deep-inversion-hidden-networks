r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0, seq_len=0,
        h_dim=0, n_layers=0, dropout=0,
        learn_rate=0.0, lr_sched_factor=0.0, lr_sched_patience=0,
    )
    # DONE: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=1024, seq_len=64,
        h_dim=256, n_layers=3, dropout=0.5,
        learn_rate=0.001, lr_sched_factor=0.1, lr_sched_patience=10,
    )    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # DONE: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "To be, or not to be"
    temperature = .001
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences instead of training on the whole text for the following reasons:
1. Running backpropagation through time on very long sequences is too computationally expensive and time demanding. 
2. Running backpropagation through time on very long sequences might lead to vanishing/exploding gradients due to
the many computations that are involved in the process.
"""

part1_q2 = r"""
**Your answer:**

It is possible because the way the rnn training algorithm works, training is done sequentially on contiguous sequences
where the last hidden state of the current iteration is used as the first hidden state of the next iteration. This 
hidden state is what allows memory to be preserved between iterations.

"""

part1_q3 = r"""
**Your answer:**

We must not shuffle the order of batches during training because as we said in our previous answer, RNN training
is done sequentially on contiguous sequences in order to preserve memory between iterations.

"""

part1_q4 = r"""
**Your answer:**
1. During sampling, we want to choose the most likely prediction with high probability. Having a low temperature
means the probabilities will be less uniform, and therefore the char with the highest score will have a significantly
higher probability of being chosen. On the other hand, during training we want our model to explore new directions, and
we do so by using a higher temperature value.
2. When $T$ is very high the probabilities vector outputted by $softmax_T$ tends to be more uniform:
$$
\frac{e^{\vec{y}/T}}{\sum_k e^{y_k/T}}
\xrightarrow[T\to\infty]{}
\frac{\vec{1}}{\sum_k 1}
$$
This means the characters will be sampled from a more uniform distribution, which means that their scores will have less
effect on their probability of being chosen. This allows the model to be less conservative and more willing to explore
new directions.
3. The lower $T$ is, the less uniform the char distribution becomes. This happens because for lower values of $T$ 
probabilities of chars with higher scores get amplified more compared to chars with lower scores. This results in a more
conservative and confident model.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0.0005,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # DONE: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=16,
        h_dim=512, z_dim=50, x_sigma2=0.5,
        learn_rate=0.0001, betas=(0.9, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The $\sigma^2$ hyperparameter tunes the regularization strength in the VAE loss. This hyperparameter allows us to
choose which term of the VAE loss (reconstruction loss term vs KL divergence loss term) we want to put more emphasis on
minimizing.
- By using smaller values of $\sigma^2$, we give more importance to the reconstruction loss term. This should
improve the quality of the reconstructed images.

- By using larger values of $\sigma^2$, we give more importance to the KL divergence loss term. This should
make the posterior $q(Z|X)$ closer to the prior $p(Z) \sim \mathcal{N}(\bb{0},\bb{I})$, and thus allows us to sample
from a known distribution to generate new samples (as explained in the following question). A value which is too high,
however, would mean that the model would probably not learn how to actually generate the samples (because the
reconstruction loss term would become negligible in this case).

"""

part2_q2 = r"""
**Your answer:**

1. The VAE loss consists of two terms:
    - The Reconstruction Loss: this term tells us how well the generated points fit the data, by calculating the squared 
    error between the original points in the instance space, and the respective points that were reconstructed from
    them by the model.
    - The KL Divergence Loss: this is a regularization term which is the divergence between the model posterior 
    $q(Z|X)$ and the prior $p(Z)$. This term can be interpreted as the information lost when $p(Z)$ is used to
    approximate $q(Z|X)$.
2. By minimizing the KL divergence loss we are minimizing the difference between $q(Z|X)$ and $p(Z)$. This means that
we are basically trying to model $q(Z|X)$ as $\mathcal{N}(\bb{0},\bb{I})$.

3. By modeling the posterior with the gaussian distribution, we are making it easier for ourselves to sample $\bb{z}$
from the latent space distribution so that with high probability $\Psi(\bb{z})$ will end up on the data manifold in the
instance space. In other words, this is what allows us to generate new instances, because otherwise we wouldn't know
where to sample from.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # DONE: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=4, z_dim=100,
        data_label=1, label_noise=0.1,
        discriminator_optimizer=dict(
            type='SGD',  # Any name in nn.optim like SGD, Adam
            lr=0.01,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            # You an add extra args for the optimizer here
        ),
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

Since a GAN consists of two separate components that are competing with each other - the Generator and the 
Discriminator, we need to calculate their loss separately, meaning to keep track of two separate computation graphs.
During training, when updating the Generator we want to maintain the gradients of the sampling process because these 
gradients will later be used to fine-tune the components of the Generator in order to make them generate better (more
convincing) samples. On the other hand, the gradients of the sampling process have nothing to do with the components of
the Discriminator (the Discriminator's components have their own gradients which have to be calculated based on the 
classification of the samples - not based on their creation). 


Thus the sampling process gradients have to be saved during the Generator updates, and are not saved during 
Discriminator updates.
"""

part3_q2 = r"""
**Your answer:**

1. No. Since a GAN consists of two components - the Generator and the Discriminator - its success as a whole depends on
both components performing well: even if the loss of the Generator is below some threshold at some point during 
training, it might be due to poor performance by the Discriminator. The Discriminator might be convinced that the 
samples generated by the Generator are real - even though in reality they are far from that. This phenomenon is pretty 
common during the first few epochs of training in which the Discriminator is still severely underfitted and thus 
tends to perform poorly. 

2. If the Generator's loss decreases, this means that the Discriminator was wrong on more samples produced by the 
Generator - i.e. classified them as 'real'. This in turn means that the Discriminator's loss must go up, but since we
know that it remained constant we must conclude that the Discriminator ability to classify real samples was improved 
(since the Discriminator's loss depends on its ability to classify both fake and real samples). Overall this indicates
that now the Discriminator tends to classify more samples as real regardless of their origin. 

"""

part3_q3 = r"""
**Your answer:**

By looking at the results it seems that the samples generated by the VAE are significantly more noisy than those 
generated by the GAN.

   https://openreview.net/pdf?id=B1ElR4cgg

VAE-based techniques learn an approximate inference mechanism that allows reuse for various auxiliary tasks, 
such as semi-supervised learning or inpainting. They do however suffer from a well- recognized issue of the maximum
 likelihood training paradigm when combined with a conditional independence assumption on the output given
  the latent variables: they tend to distribute probability mass diffusely over the data space (Theis et al., 2015).
   The direct consequence of this is that image samples from VAE-trained models tend to be blurry
   
"""

# ==============


