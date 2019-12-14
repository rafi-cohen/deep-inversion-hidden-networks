from hw2.experiments import run_experiment
import itertools as it

K = [32, 64]
L = [2, 4, 8, 16]

for k, l in it.product(K, L):
    run_experiment(run_name=f'exp1_1_L{l}_K{k}',
                   seed=42,
                   bs_train=128,
                   bs_test=None,
                   batches=100,
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
