## train model for the paper 

*Bridge Text and Knowledge using Multi-Prototype Mention Embeddding*

align.c: baseline for the paper *Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation*

MPME.c: core models for learning Multi-prototype mention embeddings, default parameters can be found in demo-mpme.sh. using skip-gram's cw.

SPME.c: for single prototype mention embeddings

MPME_cbow.c: using standard cbow for training.

selftrain.c: load pretrained word and mention embeddings, using text to induce more sense out of KB.

distance.c: given a word/mention, output nearest words or entities. take word and mention (sense) embeddings as input.