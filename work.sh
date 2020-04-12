python3 -u -m parser.work --test_data ../data/AMR/amr_2.0/test.txt.features.preproc \
               --test_batch_size 6666 \
               --load_path ../../ckpts/cleaned/ckpt_bert_nore_2_epoch126_batch55999.cleaned \
               --beam_size 8\
               --alpha 0.6\
               --max_time_step 100\
               --output_suffix _test_out
