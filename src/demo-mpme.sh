ulimit -s unlimited
ulimit -c unlimited
make
time ./MPME -train_text /home/caoyx/data/dump20170401/enwiki_cl/anchor_text_cl.dat -train_kg /home/caoyx/data/dump20170401/enwiki_cl/mono_kg_id.dat -train_anchor /home/caoyx/data/dump20170401/enwiki_cl/anchor_text_cl.dat -read_text_path /home/caoyx/data/etc/vocab/envocab/ -read_kg_path /home/caoyx/data/etc/vocab/envocab/ -output_path /home/caoyx/data/etc/exp3/envec/ -read_title_path /home/caoyx/mpme -save_iter 1 -text_sense 1 -min-count 5 -size 200 -negative 5 -sample 1e-4 -threads 63 -iter 1 -window 5
