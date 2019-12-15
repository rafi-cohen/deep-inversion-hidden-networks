from hw2.experiments import run_experiment
import itertools as it

L = [2, 4, 8]
K = [32, 64, 128, 256]

for l, k in it.product(L, K):
    run_experiment(run_name='exp1_2',
                   seed=42,
                   bs_train=128,
                   bs_test=26,
                   batches=275,
                   epochs=100,
                   early_stopping=5,
                   checkpoints=None,
                   lr=1e-3,
                   reg=1e-3,
                   filters_per_layer=[k],
                   layers_per_block=l,
                   pool_every=l,
                   hidden_dims=[1024],
                   model_type='cnn')
