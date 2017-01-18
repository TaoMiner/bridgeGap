ulimit -s unlimited
ulimit -c unlimited
make
time ./bridge -train_text /data/m1/cyx/etc/enwiki_tmp/train_text -train_kg /data/m1/cyx/etc/enwiki_tmp/train_kg -save_text_path /data/m1/cyx/etc/output/wordvec/ -save_kg_path /data/m1/cyx/etc/output/entityvec/ -output_path /data/m1/cyx/etc/output/exp1/ -min-count 5 -size 200 -negative 5 -sample 1e-4 -threads 63 -iter 1 -window 5
