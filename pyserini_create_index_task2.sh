#!/bin/bash
create_index(){
          arg1=$1
          python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator  -threads 1 -input /mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task2/task2_test_2020_copy/$arg1/  -index /mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task2/task2_test_2020_copy/$arg1/index/ -storePositions -storeDocvectors -storeRaw
      }
for sub_dir in {326..327}
do
        echo $sub_dir
        create_index $sub_dir
done
#python coliee_task1_search_index.py --train-dir /home/salthamm/data/coliee19/task1/task1_train/
# here the searching fails for too long indices - store these indices with a try and except and then try with shorter size again
#python coliee_task1_eval_index.py --train-dir /home/salthamm/data/coliee19/task1/task1_train/
# reaches recall of 0.9322899536843366 on coliee task 1 train set
