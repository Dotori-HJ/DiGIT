import glob
import json
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

# with open('data/thumos14/I3D_2stream_Pth/video_test_0000004', 'rb') as f:
#     data = pickle.load(f)
# data = torch.load('data/thumos14/I3D_2stream_Pth/video_test_0000004')
# print(data.dtype)
# root = 'data/thumos14/I3D_2stream_act_Pth'
# for path in glob.glob('data/thumos/i3d_features/*.npy'):
#     data = np.load(path)
#     data = torch.from_numpy(data).to(torch.float32)

#     name = os.path.splitext(os.path.split(path)[-1])[0]

#     torch.save(data, os.path.join(root, name))



fps = 25
frames = 16
stride = 4

result = {}

# root = 'data/thumos14/I3D_2stream_Pth'
# root = 'data/thumos14/I3D_2stream_act_Pth'
# root = 'data/features/THUMOS14/vit_g_hybrid_pt_1200e_k710_ft_norm_stride4'

root = 'data/features/ActivityNetv1.3/activitynet_internvideo2_6b_w16_s8/activitynet_6b'
gt_path = 'data/anet/anet1.3_tsp_filtered.json'
# gt_path = 'data/thumos14/th14_annotations_with_fps_duration.json'
# gt_path = 'data/activitynet/activity_net.v1-3.min.json'
# root = 'data/epic_kitchens/features'
with open(gt_path, 'rt') as f:
    gt = json.load(f)['database']

feature_lengths = []
video_names = os.listdir(root)
for video_name in tqdm(video_names):
    path = os.path.join(root, video_name)

    # features = torch.load(path)
    features = np.load(path)
    video_name = os.path.splitext(video_name)[0]
    # fps = gt[video_name]['fps']

    feature_length = len(features)
    feature_lengths.append(feature_length)
    feature_second = stride / fps * feature_length + (frames - stride) / fps
    # result[video_name[2:]] = {
    result[video_name] = {
        "feature_length": feature_length,
        "feature_second": feature_second,
        "feature_fps": round(feature_length / feature_second, 2),
        "fps": fps,
        "base_frames": frames,
        "stride": stride,
    }
print(max(feature_lengths))
# with open('data/thumos14/ft_info_vit_g_stride4.json', 'wt') as f:
with open('data/activitynet/ft_info_w16_s8.json', 'wt') as f:
    json.dump(result, f)


#
# Epic kitchen
#

# fps = 30
# frames = 32
# stride = 16

# result = {}

# # root = 'data/thumos14/I3D_2stream_act_Pth'
# root = 'data/epic_kitchens/features'

# video_names = os.listdir(root)
# for video_name in video_names:
#     path = os.path.join(root, video_name)

#     data = np.load(path)
#     features = data['feats']
#     video_name = os.path.splitext(video_name)[0]

#     feature_length = len(features)
#     feature_second = stride / fps * feature_length
#     result[video_name] = {
#         "feature_length": feature_length,
#         "feature_second": feature_second,
#         "feature_fps": feature_length / feature_second,
#         "fps": fps,
#         "base_frames": frames,
#         "stride": stride,
#     }

# # with open('data/thumos14/th14_i3d2s_act_ft_info.json', 'wt') as f:
# with open('data/epic_kitchens/ft_info.json', 'wt') as f:
#     json.dump(result, f)



# fps = 30
# frames = 16
# stride = 16

# result = {}

# root = 'data/hacs_mae_hugek700'
# # root = 'data/thumos14/I3D_2stream_act_Pth'
# gt_path = 'data/hacs/annotations/HACS_segments.json'
# # root = 'data/epic_kitchens/features'
# with open(gt_path, 'rt') as f:
#     gt = json.load(f)['database']

# video_names = os.listdir(root)
# for video_name in video_names:
#     path = os.path.join(root, video_name)

#     features = np.load(path)
#     video_name = os.path.splitext(video_name)[0][2:]
#     # fps = gt[video_name]['fps']

#     feature_length = len(features)
#     feature_second = stride / fps * feature_length + (frames - stride) / fps
#     result[video_name] = {
#         "feature_length": feature_length,
#         "feature_second": feature_second,
#         "feature_fps": round(feature_length / feature_second, 2),
#         "fps": fps,
#         "base_frames": frames,
#         "stride": stride,
#     }

# with open('data/hacs_ft_info.json', 'wt') as f:
#     json.dump(result, f)

# print(len(set(gt.keys()).intersection(result.keys())))
# print(len(result.keys()))
# print(len(video_names))