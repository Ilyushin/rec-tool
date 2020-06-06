# Tool for simplifying to perform experiments with collaborative filtering models

1. Install the tool
```shell script
chmod +x ./build_local.sh
./build_local.sh
```

2. Setting up a configuration. An example you can find in config_example.yaml

3. Running an experiments
```shell script
cf_experiments_loop --config ./config_example.yaml
```
## Config options
Config example
This config provides an option to start several experiments with some models (Embedding model, MLP, NCF, Matrix Factorization, SVD, etc.) on several datasets (Movilens/Bookcrossing/Behance/Goodreads).
There are few additional options like using different metrics for evaluation together, Grid Search on batch size parameter, number of epochs and learning rate.

```config:
  data:
    input_data:
      clear: true
      movielens:
        use: true
        type: ml-1m
        path: /tmp/cf_experiments_loop/dataset/movielens
        transformations: cf_experiments_loop.transformations.movielens.prepare_data
```

The additional datasets could be defined as follows

```
      goodreads:
        use: false
        type: goodreads
        transformations: cf_experiments_loop.transformations.goodreads.get_goodreads_data
      bookcrossing:
        use: false
        type: bookcrossing
        transformations: cf_experiments_loop.transformations.bookcrossing.bookcrossing_converting
      behance:
        use: false
        type: behance
        transformation: cf_experiments_loop.transformations.behance.behance_converting
        
```

There is option to run several models 

```        
  model:
    model: [
      cf_experiments_loop.models.embedding.embedding_model,
      cf_experiments_loop.models.mlp.mlp,
      cf_experiments_loop.models.ncf.ncf_model,
      cf_experiments_loop.models.mf.mf,
      cf_experiments_loop.models.svd.svd,
    ]
```

For model evaluation several metrics could be defined as well

```
    loss: cf_experiments_loop.losses.mean_squared_error
    metrics: [
      cf_experiments_loop.metrics.accuracy,
      cf_experiments_loop.metrics.rmse,
      cf_experiments_loop.metrics.mae
    ]
```

To start GridSearch over batch_size, epochs and learning_rate you need to define the range of these parameters using lists
```
    batch_size: [1024, 2048, 4096]
    epoch: [50, 100, 200]
    optimizers: adam
    grid_search: True
    learning_rate: 0.01
```

To save the model you need define a directory for the model. There is also an option to write results into csv file.
```
  result:
    model: /tmp/cf_experiments_loop/model/
    log: /tmp/cf_experiments_loop/log/
    results_csv: run_results.csv
    log_to_ml_flow: True
    clear: true

```
Example with the one model, goodreads dataset, one batch_size, one epoch

```
config:
  data:
    input_data:
      clear: true
      movielens:
        use: false
        type: ml-1m
        path: /tmp/cf_experiments_loop/dataset/movielens
        transformations: cf_experiments_loop.transformations.movielens.prepare_data
      goodreads:
        use: true
        type: goodreads
        transformations: cf_experiments_loop.transformations.goodreads.get_goodreads_data
      bookcrossing:
        use: false
        type: bookcrossing
        transformations: cf_experiments_loop.transformations.bookcrossing.bookcrossing_converting
      behance:
        use: false
        type: behance
        transformation: cf_experiments_loop.transformations.behance.behance_converting
  model:
    model: [cf_experiments_loop.models.ncf.ncf_model]
    loss: cf_experiments_loop.losses.mean_squared_error
    metrics: [
      cf_experiments_loop.metrics.accuracy,
      cf_experiments_loop.metrics.rmse,
      cf_experiments_loop.metrics.mae
    ]
    batch_size: [1024]
    epoch: [50]
    optimizers: adam
    grid_search: True
    learning_rate: 0.01


  result:
    model: /tmp/cf_experiments_loop/model/
    log: /tmp/cf_experiments_loop/log/
    results_csv: run_results.csv
    log_to_ml_flow: True
    clear: true
```



## Collaborative Filtering models description

| Model and paper | Examples |
| --- | :---: |
| [Variational Autoencoder for Collaborative Filtering (VAECF)](https://arxiv.org/pdf/1802.05814.pdf) | [vae.py](cf_experiments_loop/models/vae.py)
| [Singular Value Decomposition (SVD)](https://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf)| [svd.py](cf_experiments_loop/models/svd.py)
| [Matrix Factorization (MF)](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) | [mf.py](cf_experiments_loop/models/embedding.py)
| [Multi-Layer Perceptron (MLP)](https://arxiv.org/pdf/1708.05031.pdf) | [mlp.py](cf_experiments_loop/models/embedding.py)
| [Neural Matrix Factorization (NeuMF) / Neural Collaborative Filtering (NCF)](https://arxiv.org/pdf/1708.05031.pdf) | [ncf.py](cf_experiments_loop/models/ncf.py)
| [Bayesian Personalized Ranking (BPR)](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) | [bpr.py](cf_experiments_loop/models/bpr.py)
| [Weighted Matrix Factorization (WMF)](http://yifanhu.net/PUB/cf.pdf) |[wmf.py](cf_experiments_loop/models/mf.py)
| [SVD++](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp)| [svdpp.py](cf_experiments_loop/models/svdpp.py)