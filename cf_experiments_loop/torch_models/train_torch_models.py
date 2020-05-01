import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import BPR, WBPR, PMF, MF, IBPR, MMMF, NeuMF, SVD, NMF, WMF, VAECF


feedback = cornac.datasets.movielens.load_feedback(
    variant="100K", reader=cornac.data.Reader(bin_threshold=1.0)
)


def run(data):

    # Define an evaluation method to split feedback into train and test sets
    ratio_split = RatioSplit(
        data=data,
        test_size=0.1,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=True,
        seed=123
    )

    # Instantiate the most popular baseline, BPR, and WBPR models
    most_pop = cornac.models.MostPop()
    bpr = BPR(k=50, max_iter=200, learning_rate=0.001, lambda_reg=0.001, verbose=True)
    wbpr = WBPR(k=50, max_iter=200, learning_rate=0.001, lambda_reg=0.001, verbose=True)
    pmf = PMF(k=50, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)
    mf = MF(k=50, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, early_stop=True, verbose=True)
    ibpr = IBPR(k=50, verbose=True)
    mmmf = MMMF(k=50, max_iter=200, learning_rate=0.01, verbose=True)
    neumf1 = NeuMF(num_factors=8, layers=[64, 32, 16, 8], act_fn="tanh", learner="adam", num_epochs=10,
                   batch_size=256, lr=0.001, num_neg=50, seed=123)
    svd = SVD(k=50, max_iter=30, learning_rate=0.001, lambda_reg=0.02, verbose=True)
    vaecf = VAECF(k=10, autoencoder_structure=[20], act_fn="tanh", likelihood="mult", n_epochs=100, batch_size=100,
                  learning_rate=0.001, beta=1.0, seed=123, use_gpu=True, verbose=True)

    wmf = WMF(k=50, max_iter=50, learning_rate=0.001, lambda_u=0.01, lambda_v=0.01,
              verbose=True, seed=123)

    # Instantiate a NMF recommender model.
    nmf = NMF(
        k=15,
        max_iter=50,
        learning_rate=0.005,
        lambda_u=0.06,
        lambda_v=0.06,
        lambda_bu=0.02,
        lambda_bi=0.02,
        use_bias=False,
        verbose=True,
        seed=123,
    )

    gmf = cornac.models.GMF(
        num_factors=8,
        num_epochs=10,
        learner="adam",
        batch_size=256,
        lr=0.001,
        num_neg=50,
        seed=123,
    )
    mlp = cornac.models.MLP(
        layers=[64, 32, 16, 8],
        act_fn="tanh",
        learner="adam",
        num_epochs=10,
        batch_size=256,
        lr=0.001,
        num_neg=50,
        seed=123,
    )

    neumf2 = cornac.models.NeuMF(
        name="NeuMF_pretrained",
        learner="adam",
        num_epochs=10,
        batch_size=256,
        lr=0.001,
        num_neg=50,
        seed=123,
        num_factors=gmf.num_factors,
        layers=mlp.layers,
        act_fn=mlp.act_fn,
    ).pretrain(gmf, mlp)

    # Use AUC and Recall@20 for evaluation
    auc = cornac.metrics.AUC()
    rec_20 = cornac.metrics.Recall(k=100)
    ndcg = cornac.metrics.NDCG(k=50)

    # Put everything together into an experiment and run it
    cornac.Experiment(
        eval_method=ratio_split,
        models=[most_pop, bpr, wbpr, pmf, mf, ibpr, mmmf, neumf1, svd, vaecf, wmf, nmf, mlp, neumf2],
        metrics=[auc, rec_20, ndcg],
        user_based=True,
    ).run()

