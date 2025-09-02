from typing import List, Tuple

import numpy as np
from pymilvus import Collection, SearchResult


def get_milvus_predict_cate(
        cate_collection: Collection,
        sku_vector: List[np.ndarray]
) -> SearchResult:
    """
    Get the top 5 category prediction results from milvus.
    :param cate_collection: milvus category collection
    :param sku_vector: the embedding vector of the sku name to be predicted
    :return: the top 5 category prediction results
    """

    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    results = cate_collection.search(
        data=sku_vector,
        anns_field="category_embedding",
        param=search_params,
        limit=5,
        expr="",
        output_fields=["category_id"],
        **{"consistency_level": "Strong"}
    )

    return results


def cal_predict_results_acc(
        results: SearchResult,
        labels: list,
        cate_dict: dict
) -> Tuple[int, int]:
    """
    Calculate the accuracy of the category prediction results.
    :param results: the top 5 category prediction results
    :param labels: the ground truth of the category
    :param cate_dict: the category dict, key is category id, value is
        category name
    :return: the top 1 accuracy and top 3 accuracy
    """
    acc_top1, acc_top3 = 0, 0
    predict_cate_list = []
    for i, res in enumerate(results):
        result_label = [cate_dict[i] for i in res.ids]
        if labels[i] == result_label[0]:
            acc_top1 += 1
        if labels[i] in result_label[:3]:
            acc_top3 += 1

    return acc_top1, acc_top3
