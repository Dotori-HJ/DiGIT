# ------------------------------------------------------------------------
# Modified from DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from .dino import build_dino
from .digit import build

def build_model(args):
    return build(args)
