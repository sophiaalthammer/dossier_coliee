import os
import csv
import ast

# task1
mode = ['test', 'separate_para', 'overlap_ranks', 'legal_task2_dpr']
aggregated_run_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/arian/bertpli/test_all_tritanium_result_gpu0_v7qrel.txt'
    #'/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/arian/bertvanilla/test_coliee21_test_run_vbert_trained_on_coliee20.txt'
    #'/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/arian/bertpli/test_all_tritanium_result_gpu0_v7qrel.txt'
output_file = '/mnt/c/Users/salthamm/Documents/phd/paper/2021_coliee/submission_files_final/task1/run_test_bertpli2.txt'
cut_off = 7

# task2
#bm25
#aggregated_run_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25/aggregate/test/search_test_something_aggregation_overlap_scores.txt'
#output_file = '/mnt/c/Users/salthamm/Documents/phd/paper/2021_coliee/submission_files/task2/bm25/run_test_bm25.txt'
#cut_off = 1

#dpr
#aggregated_run_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/dpr/aggregate/test/search_test_something_aggregation_scores.txt'
#output_file = '/mnt/c/Users/salthamm/Documents/phd/paper/2021_coliee/submission_files/task2/dpr/run_test_dpr.txt'

#bm25+dpr
#aggregated_run_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25_dpr/aggregate/test/search_test_weighting_1_4_aggregation_scores.txt'
#output_file = '/mnt/c/Users/salthamm/Documents/phd/paper/2021_coliee/submission_files/task2/dpr/run_test_bm25_dpr.txt'

lines_dict = {}
with open(aggregated_run_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        text = line.split(' ')
        if lines_dict.get(text[0]):
            if text[2] != text[0]:
                lines_dict.get(text[0]).update({int(text[3]):text[2]})
        else:
            lines_dict.update({text[0]:{}})
            if text[2] != text[0]:
                lines_dict.get(text[0]).update({int(text[3]): text[2]})

# now order the values by keys!
sorted_run = {}
for key, value in lines_dict.items():
    sorted_list = sorted(value.items())
    sorted_candidates = []
    for candidate in sorted_list:
        sorted_candidates.append(candidate[1])
    sorted_run.update({key: sorted_candidates[:cut_off]})

# now write out in certain format
with open(output_file, 'w') as f:
    for key, value in sorted_run.items():
        for val in value:
            f.write('{} {} {}\n'.format(key, val, 'dssir1berpli'))



