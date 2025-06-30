import os
import logging
import json
import random
import os.path as osp
import numpy as np
import shutil
import copy
import time
import datetime

import torch

from util.slconfig import SLConfig

from util.segment_ops import get_time_coverage, segment_iou, segment_length

class Error(OSError):
    pass



def add_segment_occlusions(features, segment_length, max_occlusion_ratio=0.5, occlusion_value=0):
    """
    Add occlusions within a segment without covering the whole segment.

    :param features: torch.Tensor, the feature tensor to augment.
    :param segment_length: int, the length of the full segment.
    :param max_occlusion_ratio: float, the maximum ratio of the segment that can be occluded.
    :param occlusion_value: int or float, the value to set for occlusion.
    :return: torch.Tensor, the augmented feature tensor with occlusion.
    """
    # occlusions_to_add = random.randint(1, 3)  # Decide on a number of occlusions to add
    max_occlusion_length = int(segment_length * max_occlusion_ratio)
    if max_occlusion_length > 0:
        num_occlusion = random.randint(1, 3)
        for i in range(num_occlusion):
            occlusion_length = random.randint(1, max_occlusion_length)
            start = random.randint(0, segment_length - occlusion_length)
            features[start:start+occlusion_length] = occlusion_value

    return features

def add_background_occlusions(features, segments, max_occlusion_ratio=0.1, occlusion_value=0):
    """
    Add occlusions to the background (non-segment) areas of the features.

    :param features: torch.Tensor, the feature tensor to augment.
    :param segments: torch.Tensor, the segments within the features.
    :param max_occlusion_ratio: float, the maximum ratio of each background area that can be occluded.
    :param occlusion_value: int or float, the value to set for occlusion.
    :return: torch.Tensor, the augmented feature tensor with background occlusions.
    """
    total_length = features.size(1)

    # Create a mask with the same length as the features, initialized to False
    background_mask = torch.ones(total_length, dtype=torch.bool)

    # Mask out the segment areas (these are not background)
    for segment in segments:
        segment_start, segment_end = int(segment[0].item()), int(segment[1].item())
        background_mask[segment_start:segment_end] = False

    # Find indices where the background is True
    background_indices = torch.nonzero(background_mask).squeeze()

    if len(background_indices) > 0:
        # Randomly determine the number of occlusions based on the occlusion ratio
        num_occlusions = int(len(background_indices) * max_occlusion_ratio)

        # Select random indices from the background to occlude
        occlusion_indices = background_indices[random.sample(range(len(background_indices)), num_occlusions)]

        # Apply the occlusion to the selected indices
        features[:, occlusion_indices] = occlusion_value

    return features


# class RandomBoxPerturber():
#     def __init__(self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2) -> None:
#         self.noise_scale = torch.Tensor([x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale])

#     def __call__(self, refanchors: Tensor) -> Tensor:
#         nq, bs, query_dim = refanchors.shape
#         device = refanchors.device

#         noise_raw = torch.rand_like(refanchors)
#         noise_scale = self.noise_scale.to(device)[:query_dim]

#         new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
#         return new_refanchors.clamp_(0, 1)

def add_noise_to_segments(segments, noise_level):
    center = (segments[:, 0] + segments[:, 1]) / 2
    width = 0.5 * (segments[:, 1] - segments[:, 0])
    # segment_lengths = segments[:, 1] - segments[:, 0]

    # start_noise = torch.randn_like(noise_std) * noise_std
    # end_noise = torch.randn_like(noise_std) * noise_std
    center_noise = (torch.rand_like(center) * 2 - 1) * noise_level
    width_noise = (torch.rand_like(width) * 2 - 1) * noise_level


    center = center + width * center_noise
    width = width * (1 + width_noise)

    new_segments = torch.stack((center - width, center + width), dim=1)

    return new_segments

@torch.jit.script
def get_intersection(window, segments):
    l = torch.max(window[:, 0], segments[:, 0])
    r = torch.min(window[:, 1], segments[:, 1])
    inter = (r - l).clamp(min=0)  # [N,M]
    return inter

def truncate_feats(
    features,
    segments,
    labels,
    max_seq_len,
    fps,
    base_frames,
    stride,
    trunc_thr=0.5,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False,
):
    # Modified from Actionformer https://github.com/happyharrycn/actionformer_release/blob/main/libs/datasets/data_utils.py
    if len(features.shape) == 1:
        feat_len = len(features)
    else:
        feat_len = features.size(1)

    # if crop_ratio is not None:
    #     crop_len = round(random.uniform(crop_ratio[0], crop_ratio[1]) * feat_len)
    #     crop_len = max(crop_len, 16)
    #     max_seq_len = min(crop_len, max_seq_len)

    # if feat_len == max_seq_len:
    #     return features, segments, labels

    # seq_len < max_seq_len
    if feat_len <= max_seq_len:
        # do nothing
        if crop_ratio is None:
            return features, segments, labels
        # randomly crop the seq by setting max_seq_len to a value in [l, r]
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            # # corner case
            if feat_len == max_seq_len:
                return features, segments, labels


    # otherwise, deep copy the dict
    feature_duration = get_time_coverage(
        max_seq_len, fps, base_frames, stride,
    )

    # try a few times till a valid truncation with at least one action
    for _ in range(max_num_trials):
        # sample a random truncation of the video feats
        start_idx = random.randint(0, feat_len - max_seq_len)
        st = start_idx * stride / fps
        et = st + feature_duration

        # compute the intersection between the sampled window and all segments
        window = torch.tensor([[st, et]])

        # l = torch.max(window[:, 0], segments[:, 0])
        # r = torch.min(window[:, 1], segments[:, 1])
        # inter = (r - l).clamp(min=0)  # [N,M]
        inter = get_intersection(window, segments)

        segment_window = segment_length(segments)
        inter_ratio = inter / segment_window

        # only select those segments over the thresh
        valid = inter_ratio > trunc_thr  # ensure center point
        # valid = segment_window > (1 / fps)

        if no_trunc:
            # with at least one action and not truncating any actions
            seg_trunc_idx = torch.logical_and(
                (inter_ratio > 0.0), (inter_ratio < 1.0)
            )
            if (valid.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            # with at least one action
            if valid.sum().item() > 0:
                break
        else:
            # without any constraints
            break

    # feats: C x T
    # features = features[:, start_idx:start_idx + max_seq_len].clone()
    if len(features.shape) == 1:
        features = features[start_idx:start_idx + max_seq_len]
    else:
        features = features[:, start_idx:start_idx + max_seq_len]
    # segments: N x 2 in feature grids
    segments = segments[valid] - st
    # segments = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    # shift the time stamps due to truncation
    # print(segments, st)
    segments = torch.clamp(segments, 0, feature_duration)
    # labels: N
    labels = labels[valid].clone()

    return features, segments, labels



def get_feature_grid(feature_length, fps=30, window_size=16, stride=4):
    # Create feature indices: [0, 1, 2, ..., total_features - 1]
    feature_indices = torch.arange(0, feature_length)

    # Calculate the center frame index for each feature
    center_frame_indices = feature_indices * stride + window_size // 2

    # Convert the center frame indices to time in seconds
    feature_grid = center_frame_indices.float() / fps

    return feature_grid


def get_time_coverage(feature_length, fps=30, window_size=16, stride=4):
    # return stride / fps * feature_length + (window_size - stride) / fps
    return ((feature_length - 1) * stride + window_size) / fps


def load_feature(ft_path, shape=None):
    ext = os.path.splitext(ft_path)[-1]
    if ext == '.npy':
        video_df = torch.from_numpy(np.load(ft_path).T).float()
    elif ext == 'torch' or ext == '' or ext == '.pt' or ext == '.pth':
        video_df = torch.load(ft_path).T
    elif ext == '.npz':
        video_df = torch.from_numpy(np.load(ft_path)['feats'].T).float()
    elif ext == '.pkl':
        # video_df = torch.from_numpy(np.load(ft_path, allow_pickle=True))
        feats = np.load(ft_path, allow_pickle=True)
        # 1 x 2304 x T --> T x 2304
        video_df = torch.concat([feats['slow_feature'], feats['fast_feature']], dim=1).squeeze(0)#.transpose(0, 1)
    else:
        raise ValueError('unsupported feature format: {}'.format(ext))
    return video_df


def get_classes(gt):
    '''get class list from the annotation dict'''
    if 'classes' in gt:
        classes = gt['classes']
    else:
        database = gt
        all_gts = []
        for vid in database:
            all_gts += database[vid]['annotations']
        classes = list(sorted({x['label'] for x in all_gts}))
    return classes



def get_dataset_dict(video_info_path, video_anno_path, subset, mode='test', exclude_videos=None, online_slice=False, slice_len=None, ignore_empty=True, slice_overlap=0, return_id_list=False):
    '''
    Prepare a dict that contains the information of each video, such as duration, annotations.
    Args:
        video_info_path: path to the video info file in json format. This file records the length and fps of each video.
        video_anno_path: path to the ActivityNet-style video annotation in json format.
        subset: e.g. train, val, test
        mode: train (for training) or test (for inference).
        online_slice: cut videos into slices for training and testing. It should be enabled if the videos are too long.
        slice_len: length of video slices.
        ignore_empty: ignore video slices that does not contain any action instance. This should be enabled only in the training phase.
        slice_overlap: overlap ration between adjacent slices (= overlap_length / slice_len)

    Return:
        dict
    '''
    with open(video_info_path, 'rt') as f:
        video_ft_info = json.load(f)
    with open(video_anno_path, 'rt') as f:
        anno_data = json.load(video_anno_path)['database']

    video_dict, id_list = {}, {}
    cnt = 0

    video_set = set([x for x in anno_data if anno_data[x]['subset'] in subset])
    video_set = video_set.intersection(video_ft_info.keys())

    if exclude_videos is not None:
        assert isinstance(exclude_videos, (list, tuple))
        video_set = video_set.difference(exclude_videos)

    video_list = list(sorted(video_set))

    for video_name in video_list:
        # remove ambiguous instances on THUMOS14
        annotations = [x for x in anno_data[video_name]['annotations'] if x['label'] != 'Ambiguous']
        annotations = list(sorted(annotations, key=lambda x: sum(x['segment'])))

        if video_name in video_ft_info:
            # video_info records the length in snippets, duration and fps (#frames per second) of the feature/image sequence
            video_info = video_ft_info[video_name]
            # number of frames or snippets
            feature_length = int(video_info['feature_length'])
            feature_fps = video_info['feature_fps']
            feature_second = video_info['feature_second']
        else:
            continue

        video_subset = anno_data[video_name]['subset']
        # For THUMOS14, we crop video into slices of fixed length
        if online_slice:
            stride = slice_len * (1 - slice_overlap)

            if feature_length <= slice_len:
                slices = [[0, feature_length]]
            else:
                # stride * (i - 1) + slice_len <= feature_length
                # i <= (feature_length - slice_len)
                num_complete_slices = int(math.floor(
                    (feature_length / slice_len - 1) / (1 - slice_overlap) + 1))
                slices = [
                    [int(i * stride), int(i * stride) + slice_len] for i in range(num_complete_slices)]
                if (num_complete_slices - 1) * stride + slice_len < feature_length:
                    # if video_name == 'video_test_0000006':
                    #     pdb.set_trace()
                    if mode != 'train':
                        # take the last incomplete slice
                        last_slice_start = int(stride * num_complete_slices)
                    else:
                        # move left to get a complete slice.
                        # This is a historical issue. The performance might be better
                        # if we keep the same rule for training and inference
                        last_slice_start = max(0, feature_length - slice_len)
                    slices.append([last_slice_start, feature_length])
            num_kept_slice = 0
            for slice in slices:
                if 'base_frames' in video_info.keys():
                    time_slices = [
                        slice[0] * video_info['stride'] / video_info['fps'],
                        slice[1] * video_info['stride'] / video_info['fps'] + \
                            (video_info['base_frames'] - video_info['stride']) / video_info['fps'],
                    ]
                    feature_second = time_slices[1] - time_slices[0]
                    feature_fps = slice_len / feature_second
                else:
                    time_slices = [slice[0] / video_info['feature_fps'], slice[1] / video_info['feature_fps']]
                    feature_second = time_slices[1] - time_slices[0]
                    feature_fps = video_info['feature_fps']

                feature_second = time_slices[1] - time_slices[0]
                # perform integrity-based instance filtering
                valid_annotations = get_valid_anno(annotations, time_slices)

                if not ignore_empty or len(valid_annotations) >= 1:
                    # rename the video slice
                    new_vid_name = video_name + '_window_{}_{}'.format(*slice)
                    new_vid_info = {
                        'annotations': valid_annotations, 'src_vid_name': video_name,
                        'feature_fps': feature_fps, 'feature_length': slice_len,
                        'subset': subset, 'feature_second': feature_second, 'time_offset': time_slices[0],
                        'fps': video_info['fps'], 'base_frames': video_info['base_frames'], 'stride': video_info['stride'],
                    }
                    video_dict[new_vid_name] = new_vid_info
                    id_list.append(new_vid_name)
                    num_kept_slice += 1
            if num_kept_slice > 0:
                cnt += 1
        # for ActivityNet and hacs, use the full-length videos as samples
        else:
            if not ignore_empty or len(annotations) >= 1:
                # Remove incorrect annotions on ActivityNet
                valid_annotations = [x for x in annotations if x['segment'][1] - x['segment'][0] > 0.02]

                if ignore_empty and len(valid_annotations) == 0:
                    continue

                video_dict[video_name] = {
                    'src_vid_name': video_name, 'annotations': valid_annotations,
                    'feature_fps': feature_fps, 'feature_length': int(feature_length),
                    'subset': video_subset, 'feature_second': feature_second, 'time_offset': 0}
                id_list.append(video_name)
                cnt += 1
    logging.info('{} videos, {} slices'.format(cnt, len(video_dict)))
    if return_id_list:
        return video_dict, id_list
    else:
        return video_dict


def slcopytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copyfile,
             ignore_dangling_symlinks=False):
    """
    modified from shutil.copytree without copystat.

    Recursively copy a directory tree.

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied. If the file pointed by the symlink doesn't
    exist, an exception will be added in the list of errors raised in
    an Error exception at the end of the copy process.

    You can set the optional ignore_dangling_symlinks flag to true if you
    want to silence this exception. Notice that this has no effect on
    platforms that don't support os.symlink.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    The optional copy_function argument is a callable that will be used
    to copy each file. It will be called with the source path and the
    destination path as arguments. By default, copy2() is used, but any
    function that supports the same signature (like copy()) can be used.

    """
    errors = []
    if os.path.isdir(src):
        names = os.listdir(src)
        if ignore is not None:
            ignored_names = ignore(src, names)
        else:
            ignored_names = set()

        os.makedirs(dst)
        for name in names:
            if name in ignored_names:
                continue
            srcname = os.path.join(src, name)
            dstname = os.path.join(dst, name)
            try:
                if os.path.islink(srcname):
                    linkto = os.readlink(srcname)
                    if symlinks:
                        # We can't just leave it to `copy_function` because legacy
                        # code with a custom `copy_function` may rely on copytree
                        # doing the right thing.
                        os.symlink(linkto, dstname)
                    else:
                        # ignore dangling symlink if the flag is on
                        if not os.path.exists(linkto) and ignore_dangling_symlinks:
                            continue
                        # otherwise let the copy occurs. copy2 will raise an error
                        if os.path.isdir(srcname):
                            slcopytree(srcname, dstname, symlinks, ignore,
                                    copy_function)
                        else:
                            copy_function(srcname, dstname)
                elif os.path.isdir(srcname):
                    slcopytree(srcname, dstname, symlinks, ignore, copy_function)
                else:
                    # Will raise a SpecialFileError for unsupported file types
                    copy_function(srcname, dstname)
            # catch the Error from the recursive copytree so that we can
            # continue with other files
            except Error as err:
                errors.extend(err.args[0])
            except OSError as why:
                errors.append((srcname, dstname, str(why)))
    else:
        copy_function(src, dst)

    if errors:
        raise Error(errors)
    return dst

def check_and_copy(src_path, tgt_path):
    if os.path.exists(tgt_path):
        return None

    return slcopytree(src_path, tgt_path)


def remove(srcpath):
    if os.path.isdir(srcpath):
        return shutil.rmtree(srcpath)
    else:
        return os.remove(srcpath)


def preparing_dataset(pathdict, image_set, args):
    start_time = time.time()
    dataset_file = args.dataset_file
    data_static_info = SLConfig.fromfile('util/static_data_path.py')
    static_dict = data_static_info[dataset_file][image_set]

    copyfilelist = []
    for k,tgt_v in pathdict.items():
        if os.path.exists(tgt_v):
            if args.local_rank == 0:
                print("path <{}> exist. remove it!".format(tgt_v))
                remove(tgt_v)
            # continue

        if args.local_rank == 0:
            src_v = static_dict[k]
            assert isinstance(src_v, str)
            if src_v.endswith('.zip'):
                # copy
                cp_tgt_dir = os.path.dirname(tgt_v)
                filename = os.path.basename(src_v)
                cp_tgt_path = os.path.join(cp_tgt_dir, filename)
                print('Copy from <{}> to <{}>.'.format(src_v, cp_tgt_path))
                os.makedirs(cp_tgt_dir, exist_ok=True)
                check_and_copy(src_v, cp_tgt_path)

                # unzip
                import zipfile
                print("Starting unzip <{}>".format(cp_tgt_path))
                with zipfile.ZipFile(cp_tgt_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(cp_tgt_path))

                copyfilelist.append(cp_tgt_path)
                copyfilelist.append(tgt_v)
            else:
                print('Copy from <{}> to <{}>.'.format(src_v, tgt_v))
                os.makedirs(os.path.dirname(tgt_v), exist_ok=True)
                check_and_copy(src_v, tgt_v)
                copyfilelist.append(tgt_v)

    if len(copyfilelist) == 0:
        copyfilelist = None
    args.copyfilelist = copyfilelist

    if args.distributed:
        torch.distributed.barrier()
    total_time = time.time() - start_time
    if copyfilelist:
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Data copy time {}'.format(total_time_str))
    return copyfilelist


