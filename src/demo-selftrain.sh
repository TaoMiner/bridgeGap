ulimit -s unlimited
ulimit -c unlimited
make
time ./selftrain -train_text /data/m1/cyx/etc/enwiki/findmention/train_text_mention -read_word_vocab /data/m1/cyx/etc/enwiki/vocab_word.txt -read_word_vector /data/m1/cyx/etc/output/exp8/vectors_word10.dat -read_kg_vocab /data/m1/cyx/etc/enwiki/vocab_entity.txt -read_title_vocab /data/m1/cyx/etc/enwiki/vocab_title.txt -read_title_vector /data/m1/cyx/etc/output/exp8/vectors_title10.dat -read_map_file /data/m1/cyx/etc/enwiki/mapping_title -output_path /data/m1/cyx/etc/output/exp7/ -save_iter 5 -cluster_threshold 0 -size 200 -negative 5 -sample 1e-4 -threads 63 -iter 5 -max_sense_num 10 -window 5
