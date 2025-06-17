"""
Common imports for use in Jupyter notebooks and across the codebase.
Copy/paste this block into a notebook cell for convenience.
"""

# Standard library
import os
import sys
import json
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import zarr
import dask
import boto3
from botocore.exceptions import ClientError
import timm

# Project modules
from src.core.config import CFG
from src.core.data_manager import DataManager