_base_ = ['../_base_.py']

feature_folder = 'data/features/HACS/hacs_segment_internvideo2_6b_w16_s8/hacs_6b'
name_format = 'v_{}_spatial_pool_feature_6.pt'
gt_path = 'data/hacs/HACS_segments_v1.1.1_af.json'
meta_info_path = None
dataset_name = 'hacs'
num_classes = 200
binary = False
resize = False
repeat_trainset = 1
mem_cache = False
stride = 8
base_frame = 16
default_fps = 30
noise_scale = 0.0

# Evaluation
prime_metric = 'mAP_raw'
nms_mode = ['raw', 'nms']
nms_thr = 0.75
nms_sigma = 0.75
nms_multi_class = True
voting_thresh = -1
min_score = 0.001
duration_thresh = 0.1
extra_cls_path = 'data/hacs/hacs_UMTv2_6B_k710+K40_f16_frozenTuning_converted_ours.json'
# extra_cls_path = None
iou_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
display_metric_indices = [0, 5, 9]


noise_scale = 0.0
noise_scaler = 0.0
seg_noise_scale = 0.0
label_smoothing = 0.0
eval_interval = 1
temperature = 10000
normalize = False

max_seq_len = 2048
downsample_rate = 1
base_scale = 32 # level 8

eval_topk = 100
length_ratio = -1
eval_workers = None

enc_layers = 6
dec_layers = 6
num_cls_head_layers = 1
num_reg_head_layers = 3
num_feature_levels = 6
num_sampling_levels = 6
two_stage = True
emb_norm_type = 'ln'
emb_relu = True
kernel_size = 3
gc_kernel_size = 11
dc_level = 2
group_conv = True

feature_dim = 3200
hidden_dim = 256
#
set_cost_class = 1
set_cost_seg = 1
set_cost_iou = 2
cls_loss_coef = 1
seg_loss_coef = 1
iou_loss_coef = 2
enc_loss_coef = 1

lr = 1e-4
weight_decay = 0
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['sampling_offsets']
lr_linear_proj_mult = 0.1
clip_max_norm = 0.1

epochs = 25
lr_drop = 22
onecyclelr = False
multi_step_lr = False
cosine_lr = False
lr_drop_list = [45, 50]
batch_size = 16
eval_batch_size = 16
save_checkpoint_interval = 100
optimizer = 'adamw'

use_checkpoint = False

pre_norm = False
dim_feedforward = 2048
enc_dropout = 0.0
dec_dropout = 0.0
emb_dropout = 0.0
attn_dropout = 0.0
n_heads = 8
n_deform_heads = 8
max_queries = 140
query_selection_ratio = 1.0
transformer_activation = 'relu'

enc_n_points = 4
dec_n_points = 4
aux_loss = True
focal_alpha = 0.25

# for ema
use_ema = True
ema_decay = 0.9998
ema_epoch = 0