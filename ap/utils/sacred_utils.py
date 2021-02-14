import collections
from incense import ExperimentLoader
from .. import constants
from pvectorc import PVector


def log_list(name, items, ex):

    for item in items:
        ex.log_scalar(name, item)


def get_experiment_loader():

    return ExperimentLoader(
        mongo_uri=constants.MONGO_URI,
        db_name=constants.DB_NAME
    )


def execute_query(loader, query, config_keys, metric_keys):

    res = collections.defaultdict(list)
    res_db = loader.find(query)

    for exp in res_db:

        flag = False
        for metric_key in metric_keys:
            if metric_key not in exp.metrics:
                flag = True
                break

        if flag:
            continue

        config_values = []
        for key in config_keys:

            key = key.split(".")
            value = exp.config

            for part in key:
                value = getattr(value, part)

            if isinstance(value, PVector):
                value = str(value.tolist())

            config_values.append(value)

        config_values = tuple(config_values)
        res[config_values].append([exp.metrics[key].values for key in metric_keys])

    return res


def delete_query(loader, query):

    res_db = loader.find(query)
    num_deleted = 0

    for exp in res_db:
        num_deleted += 1
        exp.delete(confirmed=True)

    return num_deleted
