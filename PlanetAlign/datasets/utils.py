# 文件作用：下载内置数据集所需的 .pt 文件，已有目标文件时跳过下载。
# 数据来源：PlanetAlign/datasets/douban.py → Douban.__init__() 等构造函数传入远端 ID、目录与文件名。
# 结果去向：文件写入 root，随后由 PlanetAlign/data/dataset.py → Dataset.load_dataset() 读取。

from typing import Union, Optional
from pathlib import Path
import os
import gdown

from PlanetAlign.data import Dataset


USER_AGENT = 'PlanetAlign'


# 功能：展开目录、创建目标路径并调用 gdown.download()，下载动作不包含图数据解析。
def download_file_from_google_drive(
        remote_file_id: str,
        root: Union[str, Path],
        save_filename: Optional[Union[str, Path]] = None):
    r"""Download a file from Google Drive."""

    root = os.path.expanduser(root)
    if not save_filename:
        save_filename = remote_file_id
    fpath = os.fspath(os.path.join(root, save_filename))

    fname = save_filename.split('.')[0]
    os.makedirs(root, exist_ok=True)
    if os.path.isfile(fpath):
        print(f'Dataset "{fname}" is already downloaded')
        return
    else:
        print(f'Downloading dataset "{fname}"...')

    gdown.download(id=remote_file_id, output=fpath, quiet=False, user_agent=USER_AGENT)


if __name__ == '__main__':
    # Download the Phone-Email dataset
    download_file_from_google_drive(
        remote_file_id='13BklpBEFjT73Xk8H-daPGzBx-xFbpO0P',
        save_filename='phone_email.pt',
        root='../../datasets')

    phone_email = Dataset(root='../../datasets', name='phone_email', train_ratio=0.2)
    print(phone_email)

    # # Download the Douban dataset
    # download_file_from_google_drive(
    #     remote_file_id='1J5s9F9zv1Z6L6z4P3rWVlqfF7QvZ9w7R',
    #     root='../../datasets')
