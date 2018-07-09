ulimit -s unlimited
ulimit -c unlimited
make
time ./SPME -train_text /data/m1/cyx/etc/enwiki/train_anchor_cl -train_kg /data/m1/cyx/etc/enwiki/train_kg_cl -train_anchor /data/m1/cyx/etc/enwiki/train_anchor_cl -read_text_path /data/m1/cyx/etc/enwiki/ -read_kg_path /data/m1/cyx/etc/enwiki/ -output_path /data/m1/cyx/etc/output/exp12/ -read_title_path /data/m1/cyx/etc/enwiki/ -save_iter 10 -text_sense 1 -min-count 5 -size 200 -negative 5 -sample 1e-4 -threads 63 -iter 10 -window 5
