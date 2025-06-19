# hi
------
1. `finprocess.py` -> final combined process to get `tsv` files' rank and grade
2. `m3withtsv.py`, `mrerkerwithtsv.py`, `tsvprocesser.py`(which is the same as `qwenrerker.py`) -> files to do the func which seperates (1.)
3. `read.py` ->  the first file in whole folder which is a standard form of calling model(bge-m3)
4. `rerank_bge.py` & `rerank_bge.py` -> next 2 files which do the same thing
5. `S2summary.py` -> calculate the final output from (2.)
6. `test2.py` -> obviously the same as (3.)