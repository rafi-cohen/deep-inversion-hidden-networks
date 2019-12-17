from hw2.experiments import run_experiment
import itertools as it

K = [32, 64, 128]
L = [2, 3, 6, 9, 12]

for l in L:
    run_experiment(run_name=f'exp2',
                   seed=42,
                   bs_train=128,
                   bs_test=26,
                   batches=275,
                   epochs=100,
                   early_stopping=5,
                   checkpoints=None,
                   lr=1e-3,
                   reg=1e-4,
                   filters_per_layer=K,
                   layers_per_block=l,
                   pool_every=l,
                   hidden_dims=[None],
                   model_type='ycn')
