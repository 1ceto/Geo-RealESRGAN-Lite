# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models
import inspect
from realesrgan.archs import srvgg_arch
print("使用的 SRVGGNetCompact 定义路径：", inspect.getfile(srvgg_arch.SRVGGNetCompact))

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
