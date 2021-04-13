#!/bin/bash
conda activate bert-pli
create_index(){
          arg1=$1
          python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator  -threads 1 -input /home/salthamm/data/coliee19/task1/task1_train/$arg1/  -index /home/salthamm/data/coliee19/task1/task1_train/$arg1/index/ -storePositions -storeDocvectors -storeRaw
      }
for sub_dir in {200..285}
do
        echo $sub_dir
        create_index $sub_dir
done
