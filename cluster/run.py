from cluster import Cluster
from cluster.utils import trans_same_words


def get(data: list[str], metric: str="euclidean", temperature: float=0) -> list[str]:
    """
    Get cluster result words from `data`
    :param data: data to get cluster result from
    :param metric: metric to calculate
    :param temperature: temperature to judge cluster result, remove those bad cluster words with low score
    :return: list of cluster result words
    """
    data = data * 2
    res = Cluster(data, metric=metric).run()
    res = [r[0][:1] for r in res if r[1] >= temperature]
    res = [[i[0] for i in r] for r in res]
    res = [i for i, _ in {" ".join(r): 1 for r in res}.items()]
    res = trans_same_words(res)
    return res