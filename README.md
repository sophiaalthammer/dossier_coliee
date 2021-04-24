# @COLIEE 2021: Leveraging dense retrieval and summarization re-ranking for case law retrieval (Dossier Team)
Sophia Althammer, Arian Askari, Suzan Verberne, Allan Hanbury

This repository contains the code for Dossier team which is the third team at COLIEE 2021 for case law retrieval (task 1), and fifth team in Legal Case Entailment (task 2).

### Structure

The repository is structured as follows:
- `dpr/`: The Dense Passage Retrieval (DPR) implementation which is based on the facebook research DPR repository \([github](https://github.com/facebookresearch/DPR)).
- `cedr/`: Vanilla Bert implementation which is baed on the CEDR: Contextualized Embeddings for Document Ranking implementation \([github](https://github.com/Georgetown-IR-Lab/cedr)).
- `summarizer/` contains the notebook for generating summary for coliee'21 caselaws using Longformer Encoder-Decorder (LED) model
- `runs/`:  all the final submitted run files, containing noticed cases ranked by (bm25), and proposed method (bm25+lawdpr), vanilla bert for task1.

### Data
Please visit [COLIEE 2021](https://sites.ualberta.ca/~rabelo/COLIEE2021/) to apply for the whole dataset. 

Please email a.askari@liacs.leidenuniv.nl for the checkpoint of LED summarizer (fine-tuned on COLIEE'18 summaries). 

## Contact
If you have any questions, please email sophia.althammer@tuwien.ac.at or a.askari@liacs.leidenuniv.nl. 