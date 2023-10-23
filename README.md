# Brant: Foundation Model for Intracranial Neural Signal
Brant is a foundation model in the field of intracranial recordings, which learns powerful representations of intracranial neural signals.

### Source Code

The source code of Brant is provided in `Brant_src`.

* `Brant_src/pretrain` contains the pre-training code of Brant.
* `Brant_src/train1.py`, `Brant_src/evaluate1.py` are the code for the seizure detection task.
* `Brant_src/train2.py`, `Brant_src/evaluate2.py` are the code for short- and long-term, frequency-phase forecasting tasks.
* `Brant_src/train3.py`, `Brant_src/evaluate3.py` are the code for the imputation task.

### Pre-trained weights

* We also release the [pre-trained weights of Brant](https://drive.google.com/file/d/1QzxTNBvgcJBRxa8W2mNq2Tj967GtlDLF/view?usp=sharing ) 

* You can find the usage of the pre-trained embeddings in  `Brant_src/utils.py`: `get_emb()`


### Disclaimer
The pre-training data for the Brant model was collected during routine treatment procedure of epilepsy patients from a hospital, and is intended solely for medical or research use. The pre-trained weights of Brant are released only for medical or research purposes and must not be subjected to any form of misuse. 
