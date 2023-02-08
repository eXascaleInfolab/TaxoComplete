import pdb
import time
import torch
import numpy as np
import itertools
import re
from sentence_transformers import util

from pathlib import Path

def save_results(saving_path,all_targets, all_predictions,name_file):
    Path(saving_path).mkdir(parents=True, exist_ok=True)
    f = open(saving_path + name_file + ".csv", "a")
    line_metric = "\nprec@1,prec@5,prec@10,recall@1,recall@5,recall@10,MR,MRR\n"
    f.write(line_metric)
    relevance = get_relevance(all_targets, all_predictions)
    prec1, prec5, prec10 = compute_precision(relevance, 1), compute_precision(relevance, 5), compute_precision(
        relevance, 10)
    rec1, rec5, rec10 = compute_recall(relevance, 1), compute_recall(relevance, 5), compute_recall(relevance,
                                                                                                            10)
    mr = micro_mr(relevance)
    mrr = compute_mrr(relevance)
    line = "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(prec1, prec5, prec10, rec1, rec5, rec10,
                                                                            mr,mrr)
    f.write(line)
    f.close()
    print(line_metric,line)
    return line_metric,line


def compute_prediction(edges,leaf_node, queries,corpus_embeddings,model,node_list,node2positions,corpusId2nodeId):
    top_k = len(corpus_embeddings)
    all_targets = []
    all_predictions = []
    all_scores = []
    all_edges_scores,edges_prediction = [],[]
    edges_2darray = np.array([*list(edges)])
    parent = edges_2darray[:,0]
    children = edges_2darray[:,1]
    for idx, query in enumerate(queries):
        try:
            query_id = node_list[idx]
            target_positions = node2positions[query_id]
            all_targets.append(target_positions)
            question_embedding = model.encode(query, convert_to_tensor=True)
            hits_score = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
            hits = [corpusId2nodeId[hit['corpus_id']] for hit in
                    hits_score[0]]  # Get the hits for the first query
            scores = [hit['score'] for hit in hits_score[0]]
            hits.append(leaf_node)
            scores.append(2)
            scores_arr = np.array(scores)
            ind_parents = np.where(hits==parent[:,None])[1]
            ind_child = np.where(hits==children[:,None])[1]
            scores_2darray = np.append([scores_arr[ind_parents]],[scores_arr[ind_child]],axis=0).T
            args_leaf = np.where(scores_2darray[:,1]==2)
            scores_2darray[args_leaf,1] = scores_2darray[args_leaf,0]
            # for id_x,x in enumerate(query_pred):
            #     scores_2darray[np.where(edges_2darray==x)]=scores_pred[id_x]
            scores_mean = scores_2darray.mean(axis=1)
            sorting_args = np.argsort(scores_mean)[::-1]
            edges_prediction.append(edges_2darray[sorting_args,:])
            all_edges_scores.append(scores_mean[sorting_args])
            all_predictions.append(hits)
            all_scores.append(scores)
        except:
            pdb.set_trace()
    return all_targets, all_predictions, all_scores, edges_prediction, all_edges_scores

def get_relevance(all_target, pred_pos):
    relevance = []
    pred_pos_np = np.array(pred_pos)
    for idx, target_parents in enumerate(all_target):
        relevance.append([0]*len(pred_pos[idx]))
        for (parent,child) in target_parents:
            identify_idx = np.where((pred_pos_np[idx] == (parent,child)).all(axis=1))[0]
            if len(identify_idx)>0:
                posIdx = identify_idx[0]
            else:
                posIdx = np.where(pred_pos_np[idx] == (parent,child))[0][0]  
            relevance[idx][posIdx] = 1
    return np.array(relevance)

def micro_mr(relevance):
    ranks = [np.nonzero(t)[0] for t in relevance]
    ranks_l =[elm[0] for elm in ranks]
    micro_mr = sum(ranks_l)/len(ranks_l)
    return micro_mr

def compute_recall(relevance, r=10):
    true_position_in_top_r = np.any(relevance[:, :r], axis=1)
    return np.mean(true_position_in_top_r)

def compute_precision(relevance, r=10):
    true_position_in_top_r = np.any(relevance[:, :r], axis=1)
    return 1.0 * np.sum(true_position_in_top_r) / (len(true_position_in_top_r)*r)


def compute_mrr(relevance, r=10):
    """Compute the mean reciprocal rank of a set of queries.

    relevance is a numpy matrix; each row contains the "relevance"
    of the predictions (= 0 or 1) made for each query.

    predictions are ranked in decreasing order of relevence.
    relevance[:, :15] are the top 15 most relevent predictions.

    The first non-zero entry of each row is the lowest ranked correct
    prediction. The reciprocal of this rank is the reciprocal rank.
    The mean of the reciprocal rank over all queries is returned as
    a percentage.

    Example:

        relevance = [[0, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 0, 0]]

        ranks = [[2], [1], []]  # this is 0-based

        mrrs = [1/(2+1), 1/(1+1), 0] = [1/3, 1/2, 0]
    """
    ranks = [np.nonzero(t)[0] for t in relevance[:, :]]
    mrrs = [1.0/(rank[0] + 1) if len(rank) > 0 else 0.0
            for rank in ranks]
    return 100.0 * np.mean(mrrs)