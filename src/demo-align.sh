ulimit -s unlimited
ulimit -c unlimited
make
time ./align -train_text /data/m1/cyx/etc/enwiki/train_text_cl -train_kg /data/m1/cyx/etc/enwiki/train_kg_cl -train_anchor /data/m1/cyx/etc/enwiki/train_anchor_cl -read_vocab_path /data/m1/cyx/etc/enwiki/ -output_path /data/m1/cyx/etc/output/exp14/ -binary 1 -min-count 5 -cw 0 -sg 1 -size 200 -negative 5 -sample 1e-4 -threads 63 -iter 10 -window 5
