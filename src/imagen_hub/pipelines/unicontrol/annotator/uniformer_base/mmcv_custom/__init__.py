'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
 * Modified from ControlNet repo: https://github.com/lllyasviel/ControlNet
 * Copyright (c) 2023 Lvmin Zhang and Maneesh Agrawala
 * Modified from MMCV repo: From https://github.com/open-mmlab/mmcv
 * Copyright (c) OpenMMLab. All rights reserved.
'''

# -*- coding: utf-8 -*-

from .checkpoint import load_checkpoint

__all__ = ['load_checkpoint']
