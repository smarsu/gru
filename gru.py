# Copyright (c) 2020 smarsufan. All Rights Reserved.

import torch
import numpy as np
import ctypes

libbleu = ctypes.cdll.LoadLibrary('libgru.so')
