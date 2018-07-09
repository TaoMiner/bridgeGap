WikiExtractor.py: extract wiki_id, train_text. (wiki_outlink less than yan zhang's version, redirect_cl doesn't work).

cleanwiki.py: clean train_text and wiki_outlink, construct train_anchor, count each entity's mention names in wikipedia anchors.

readWiki.py: from wiki_title and redirect title to construct **umambiguous** lower-cased mention-names and entity pairs, and wiki_title_cl, wiki_title_ignore to exclude 'Category:1909 treaties'

self_preprocess.py: to clean giga corpus

statistics.py: count the size of mention name vocab, total anchor number and the size of entity vocabulary of wikipedia anchors.

-------

mapping_rule.py: construct mention-title mappings, used for find mention via ac automachine.

mapping.py: construct mention-entity pairs. each mention refers to multiple entities.

conll_vocab.py: build conll dataset's word, mention and entity vocab