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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q4 = r"""
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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
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
        batch_size=64, z_dim=100,
        data_label=1, label_noise=0.2,
        discriminator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            lr=0.001,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            lr=0.001,
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


