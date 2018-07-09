## Evaluating word, entity and mention embeddings learned in MPME

senseLinking.py: unsupervised entity disambiguation model using aida datesets.

word.py, entity.py and title.py: load embeddings from MPME

map.py: extract mention names from wiki title, lower cased titles and redirected titles, and anchor mention names.

features.py, featuresSMPE.py: generate features for each pair of mention and candidate entity. using aida datesets and MPME/SMPE learned embeddings.

candidate.py: construct mention's candidate entities

evalEntity.py: evaluate entity relatedness using dextor datasets

evalConll.py: supervised entity disambiguation model using aida dataset and features from features.py

evalWordSim353.py, evalSCWS.py: evaluate word embeddings on those two datasets.

compute-accuracy.c: evaluate word embedding via word analogy reasoning task.