## training all the parameter from scratch
job_name="YOUR JOB NAME"
DATA_PATH="YOUR DATA PATH"
python -m torch.distributed.launch --nproc_per_node=1  --use-env --master_port 29572 \
    vpt_aug_xclip.py --do_train --num_thread_reader=1 \
    --lr 1e-4 --batch_size=64  --batch_size_val 40 \
    --epochs=5  --n_display=10 \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/VideoData \ 
    --output_dir ckpts_dsw/${job_name} \
    --max_words 32 --max_frames 12 \
    --datatype vlnoops \
    --feature_framerate 1 --coef_lr 1e-3 \
    --vpt_lr 1e-2 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqVptTransf \
    --loss_func maxcol_phrase_vpt \
    --pretrained_clip_name ViT-B/32 \




# # # # # #extracte feature
job_name="YOUR JOB NAME"
DATA_PATH="YOUR DATA PATH"
python -m torch.distributed.launch --nproc_per_node=1 --use-env  --master_port 29577 \
    exFeat_xclip_vpt.py --do_eval --num_thread_reader=8 \
    --lr 1e-4 --batch_size=8  --batch_size_val 40 \
    --epochs=5  --n_display=10 \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/VideoData \
    --output_dir ckpts_dsw/${job_name} \
    --max_words 32 --max_frames 12 \
    --datatype vlnoops \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqVptTransf \
    --use_vsiual_prompt True \
    --pretrained_clip_name ViT-B/32 \
    --save_feat_name ${job_name}_vp_ep5 \
    --save_feat_dir ${DATA_PATH}/xclip_feats \
    --init_model ckpts_dsw/${job_name}/pytorch_model.bin.4


# # # # # #test 4 pos
# job_name="oops_vpt_all_parameter_1e3_vptlr1e2"
job_name="YOUR JOB NAME"
DATA_PATH="YOUR DATA PATH"
python test_oops_negative_query_aug.py --save_dir ${DATA_PATH}/${job_name}_ep5_vp \
--visual_path ${DATA_PATH}/xclip_feats/${job_name}_vp_ep5_vis_feat.pkl \
--visual_mask_path ${DATA_PATH}/xclip_feats/${job_name}_vp_ep5_mask.pkl \
--test_model ckpts_dsw/${job_name}/pytorch_model.bin.4 \
--test_id_path ${DATA_PATH}/vln_oops_val.txt



