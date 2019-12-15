from hw2.experiments import run_experiment
import itertools as it

K = [64, 128, 256]
L = [1, 2, 3, 4]

for l in L:
    run_experiment(run_name='exp1_3',
                   seed=42,
                   bs_train=128,
                   bs_test=26,
                   batches=275,
                   epochs=100,
                   early_stopping=5,
                   checkpoints=None,
                   lr=1e-3,
                   reg=1e-3,
                   filters_per_layer=K,
                   layers_per_block=l,
                   pool_every=l,
                   hidden_dims=[1024],
                   model_type='cnn')
