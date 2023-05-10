# BPT: A Foundation Model for Intracranial Neural Signal
BPT is a foundation moded in the field of intracranial recordings, which learns powerful representations of intracranial neural signals.

### Source Code

The source code of BPT is provided in `BPT_src`.

* `BPT_src/pretrain` contains the pre-training code of BPT.
* `BPT_src/train1.py`, `BPT_src/evaluate1.py` are the code for the seizure detection task.
* `BPT_src/train2.py`, `BPT_src/evaluate2.py` are the code for short- and long-term, frequency-phase forecasting tasks.
* `BPT_src/train3.py`, `BPT_src/evaluate3.py` are the code for the imputation task.

### Pre-trained weights

* We also release the [pre-trained weights of BPT](https://drive.google.com/file/d/1QzxTNBvgcJBRxa8W2mNq2Tj967GtlDLF/view?usp=sharing ) 

* You can find the usage of the pre-trained embeddings in  `BPT_src/utils.py`: `get_emb()`


