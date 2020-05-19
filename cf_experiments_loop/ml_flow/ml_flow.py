import mlflow

mlflow.set_tracking_uri("http://10.20.2.3:8000")


def log_to_mlflow(project_name: str = None, group_name: str = None, params: dict = None, metrics: dict = None,
                  tags: dict = None, artifacts: list = []):
    """Log parameters and metrics to MLFlow
    If project with provided name does not exist - it will be created
    If experiments group with provided name does not exist - it will be created

    Parameters:
    ----------
    project_name: str
        Name of a project to log params and metrics for.
        Example: stt, tts, recommendations, etc.

    group_name: str
        Group, under which this experiment should be logged. You should understand what experiments in this group are about just by looking at the group name.
        Example: logreg, deepspeech, catboost grid search, etc.

    params: dict:
        Hyperparams used during current experiment.
        Example: {"random_seed": 1, "learning_rate": 0.01, "max_depth": 10}

    metrics: dict
        Performance metrics.
        Example: {"running_time": 117.08, "rouge": 0.51}

    tags: dict:
        Any useful tag you can think of and which is not falling under hyperparameters category.
        Example: {"dataset": "librispeech", "duration": "40h"}, {"features": { ... }}, etc.

    artifacts(list[str]):
        List of paths to any file that you think is worth saving. The file should have a reasonable size which should not exceed 20-50MB
        Example: path to model, test dataset, dataset sample, etc.

    Returns:
    ----------
    None
    """
    mlflow.set_experiment(project_name)
    project_id = mlflow.get_experiment_by_name(project_name).experiment_id
    group = mlflow.search_runs(experiment_ids=project_id, filter_string=f"tags.`mlflow.runName`='{group_name}'",
                               run_view_type=1)

    if group.empty:
        mlflow.start_run(experiment_id=project_id, run_name=group_name, nested=False)
    else:
        mlflow.start_run(run_id=group["run_id"][0], nested=False)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.set_tags(tags)
        mlflow.log_metrics(metrics)
        for artifact in artifacts:
            mlflow.log_artifact(artifact)
    mlflow.end_run()
