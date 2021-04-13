import os
import argparse
import jsonlines
import json
import numpy as np
import pickle
import csv
from eval.eval_bm25_coliee2021 import read_label_file, ranking_eval, aggregate_run_overlap, aggregate_run_interleave, aggregate_run_ranks_overlap, aggregate_run_mean_score


def read_run_whole_doc(pred_dir: str, scores='ranks'):
    with open(pred_dir, 'r') as json_file:
        pred = json.load(json_file)

    pred_dict = {}
    for question in pred:
        question_id = question.get('answers')[0].split('_')[0]
        pred_list = {}
        i = 0
        for predition in question.get('ctxs'):
            if scores == 'scores':
                pred_list.update({predition.get('id').split('_0')[0]: float(predition.get('score'))})
            else:
                pred_list.update({predition.get('id').split('_0')[0]: len(question.get('ctxs')) - i})
                i += 1
        pred_dict.update({question_id: pred_list})
    return pred_dict


def read_run_separate(pred_dir: str, scores='ranks'):
    with open(pred_dir, 'r') as json_file:
        pred = json.load(json_file)

    pred_dict = {}
    for question in pred:
        question_id = question.get('answers')[0].split('_')[0]
        question_para_id = question.get('answers')[0].split('_')[1]
        pred_list = {}
        i = 0
        for prediction in question.get('ctxs'):
            if scores == 'scores':
                pred_list.update({prediction.get('id'): float(prediction.get('score'))})
            else:
                pred_list.update({prediction.get('id'): len(question.get('ctxs')) - i})
                i += 1
        if pred_dict.get(question_id):
            pred_dict.get(question_id).update({question_para_id: pred_list})
        else:
            pred_dict.update({question_id: {}})
            pred_dict.get(question_id).update({question_para_id: pred_list})
    return pred_dict


def read_run_separate_aggregate(pred_dir: str, aggregation='interleave', scores='ranks'):
    # geh in den bm25 folder, lies in dokument und query: dann dict {query: {top 1000}}
    if aggregation == 'overlap_scores' or aggregation == 'mean_scores':
        scores = 'scores'

    run = read_run_separate(pred_dir, scores)

    # now i need an aggregation function here, different choices
    if aggregation == 'overlap_docs':
        # now aggregate according to the overlap of the docs in the paragraphs!
        print('aggregate overlapping docs')
        run_aggregated = aggregate_run_overlap(run)
    elif aggregation == 'interleave':
        print('aggregate interleave docs')
        run_aggregated = aggregate_run_interleave(run)
    elif aggregation == 'overlap_ranks':
        # now aggregate according to the overlap of the docs in the paragraphs!
        run_aggregated = aggregate_run_ranks_overlap(run)
    elif aggregation == 'overlap_scores':
        # now aggregate according to the overlap of the docs in the paragraphs!
        run_aggregated = aggregate_run_ranks_overlap(run)
    elif aggregation == 'mean_scores':
        run_aggregated = aggregate_run_mean_score(run)
    else:
        print('no aggregation but why?')
        run_aggregated = run
    return run_aggregated


def eval_ranking_dpr(label_file, pred_dir, output_dir, output_file, aggregation='interleave', scores='ranks'):

    qrels = read_label_file(label_file)
    if 'separate' in pred_dir:
        run = read_run_separate_aggregate(pred_dir, aggregation, scores)
    else:
        run = read_run_whole_doc(pred_dir, scores)

    ranking_eval(qrels, run, output_dir, output_file)


def eval_ranking_overall_recall(label_file, pred_dir, output_dir, mode, aggregation='interleave'):
    qrels = read_label_file(label_file)
    if 'separate' in pred_dir:
        run = read_run_separate_aggregate(pred_dir, aggregation)
    else:
        run = read_run_whole_doc(pred_dir)

    # overall recall
    rec_per_topic = []
    for key, value in qrels.items():
        rel_run = run.get(key)
        value_list = [val for val in value.keys()]
        rel_run_list = [rel for rel in rel_run.keys()]
        rec_per_topic.append(len(list(set(value_list) & set(rel_run_list))) / len(value_list))

    print('Overall recall of {} is {}'.format(''.join([mode[0], mode[1]]), np.mean(rec_per_topic)))

    with open(os.path.join(output_dir, 'eval_recall_overall_{}_{}.txt'.format(mode[0], mode[1])), 'w') as output:
        output.write('Overall recall of {} is {}'.format(''.join([mode[0], mode[1]]), np.mean(rec_per_topic)))


if __name__ == "__main__":
    #
    # config
    #
    #parser = argparse.ArgumentParser()

    #parser.add_argument('--corpus-dir', action='store', dest='corpus_dir',
    #                    help='corpus directory location', required=True)
    # parser.add_argument('--output-dir', action='store', dest='output_dir',
    #                    help='output directory location', required=True)
    # parser.add_argument('--label-file-train', action='store', dest='label_file',
    #                    help='label file train', required=True)
    # parser.add_argument('--label-file-test', action='store', dest='label_file_test',
    #                    help='label file test', required=True)

    #args = parser.parse_args()

    #mode = ['val', 'whole_doc', 'overlap_doc', 'bs_22_epoch_40_lr_1e05']

    #mode = ['test', 'separate_para', 'interleave']

    def eval_mode(mode):
        label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_wo_val_labels.json'
        label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'

        #pred_dir = '/mnt/c/Users/salthamm/Documents/coding/DPR/data/coliee2020_task1/rawdpr/output_rawdpr/{}/{}_{}_top1000.json'.format(mode[0], mode[0], mode[1])
        #output_dir = '/mnt/c/Users/salthamm/Documents/coding/DPR/data/coliee2020_task1/rawdpr/eval_rawdpr/{}'.format(mode[0])
        pred_dir = '/mnt/c/Users/salthamm/Documents/coding/DPR/data/coliee2021_task1/{}/output/{}/{}_{}_top1000.json'.format(mode[3],mode[0], mode[0], mode[1])
        output_dir = '/mnt/c/Users/salthamm/Documents/coding/DPR/data/coliee2021_task1/{}/eval/{}'.format(mode[3], mode[0])
        output_file = 'eval_dpr_{}_{}_{}_aggregation_{}.txt'.format(mode[3], mode[0], mode[1], mode[2])

        # evaluate retrieval of DPR
        if mode[0] == 'val':
            eval_ranking_dpr(label_file_val, pred_dir, output_dir, output_file, mode[2])
        elif mode[0] == 'train':
            eval_ranking_dpr(label_file, pred_dir, output_dir, output_file, mode[2])

        # evaluate overall recall: better for separate retrieval -> then aggregation function needs to be changed!
        #if mode[0] == 'val':
        #    eval_ranking_overall_recall(label_file_val, pred_dir, output_dir, mode)
        #elif mode[0] == 'train':
        #    eval_ranking_overall_recall(label_file, pred_dir, output_dir, mode)

    def eval_one_model(model_dir):
        #eval_mode(['val', 'whole_doc', 'overlap_docs', model_dir])
        #eval_mode(['val', 'separate_para', 'overlap_docs', model_dir])

        eval_mode(['train', 'whole_doc', 'overlap_docs', model_dir])
        #eval_mode(['train', 'separate_para', 'overlap_docs', model_dir])

        #eval_mode(['val', 'separate_para', 'overlap_ranks', model_dir])
        #eval_mode(['val', 'separate_para', 'overlap_scores', model_dir])
        #eval_mode(['train', 'separate_para', 'overlap_ranks', model_dir])
        #eval_mode(['train', 'separate_para', 'overlap_scores', model_dir])


    #eval_one_model('standard_dpr')
    eval_one_model('legal_task2_dpr')
