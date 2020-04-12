python3 -u -m parser.work --test_data data/AMR/amr_2.0_reca/test.txt.features.preproc \
               --test_batch_size 6666 \
               --load_path ../../ckpts/ckpt_bert_2_epoch103_batch54999 \
               --beam_size 8\
               --alpha 0.6\
               --max_time_step 100\
               --output_suffix _test_out
