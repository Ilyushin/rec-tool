# Tool for simplifying to perform experiments with collaborative filtering models

1. Install the tool
```shell script
chmod +x ./build_local.sh
./build_local.sh
```

2. Setting up a configuration. An example you can find in config_example.yaml

3. Running an experiments
```shell script
rec_tool --config ./config_example.yaml
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
        path: /tmp/rec_tool/dataset/movielens
        transformations: rec_tool.transformations.movielens.prepare_data
```

The additional datasets could be defined as follows

```
      goodreads:
        use: false
        type: goodreads
        transformations: rec_tool.transformations.goodreads.get_goodreads_data
      bookcrossing:
        use: false
        type: bookcrossing
        transformations: rec_tool.transformations.bookcrossing.bookcrossing_converting
      behance:
        use: false
        type: behance
        transformation: rec_tool.transformations.behance.behance_converting
        
```

There is option to run several models 

```        
  model:
    model: [
      rec_tool.models.embedding.embedding_model,
      rec_tool.models.mlp.mlp,
      rec_tool.models.ncf.ncf_model,
      rec_tool.models.mf.mf,
      rec_tool.models.svd.svd,
    ]
```

For model evaluation several metrics could be defined as well

```
    loss: rec_tool.losses.mean_squared_error
    metrics: [
      rec_tool.metrics.accuracy,
      rec_tool.metrics.rmse,
      rec_tool.metrics.mae
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
    model: /tmp/rec_tool/model/
    log: /tmp/rec_tool/log/
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
        path: /tmp/rec_tool/dataset/movielens
        transformations: rec_tool.transformations.movielens.prepare_data
      goodreads:
        use: true
        type: goodreads
        transformations: rec_tool.transformations.goodreads.get_goodreads_data
      bookcrossing:
        use: false
        type: bookcrossing
        transformations: rec_tool.transformations.bookcrossing.bookcrossing_converting
      behance:
        use: false
        type: behance
        transformation: rec_tool.transformations.behance.behance_converting
  model:
    model: [rec_tool.models.ncf.ncf_model]
    loss: rec_tool.losses.mean_squared_error
    metrics: [
      rec_tool.metrics.accuracy,
      rec_tool.metrics.rmse,
      rec_tool.metrics.mae
    ]
    batch_size: [1024]
    epoch: [50]
    optimizers: adam
    grid_search: True
    learning_rate: 0.01


  result:
    model: /tmp/rec_tool/model/
    log: /tmp/rec_tool/log/
    results_csv: run_results.csv
    log_to_ml_flow: True
    clear: true
```



## Collaborative Filtering models description

| Model and paper | Examples |
| --- | :---: |
| [Variational Autoencoder for Collaborative Filtering (VAECF)](https://arxiv.org/pdf/1802.05814.pdf) | [vae.py](rec_tool/models/vae.py)
| [Singular Value Decomposition (SVD)](https://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf)| [svd.py](rec_tool/models/svd.py)
| [Matrix Factorization (MF)](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) | [mf.py](rec_tool/models/embedding.py)
| [Multi-Layer Perceptron (MLP)](https://arxiv.org/pdf/1708.05031.pdf) | [mlp.py](rec_tool/models/embedding.py)
| [Neural Matrix Factorization (NeuMF) / Neural Collaborative Filtering (NCF)](https://arxiv.org/pdf/1708.05031.pdf) | [ncf.py](rec_tool/models/ncf.py)
| [Bayesian Personalized Ranking (BPR)](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) | [bpr.py](rec_tool/models/bpr.py)
| [Weighted Matrix Factorization (WMF)](http://yifanhu.net/PUB/cf.pdf) |[wmf.py](rec_tool/models/mf.py)
| [SVD++](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp)| [svdpp.py](rec_tool/models/svdpp.py)