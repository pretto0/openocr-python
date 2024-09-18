import os
import tarfile
import subprocess

def ensure_files_exist(file_path, tar_file_path, url):
    current_dir = os.path.dirname(__file__)
    if not os.path.exists(file_path):
        print(f'{file_path} 不存在，正在下载...')

        subprocess.run(['wget', '-O', tar_file_path, url], check=True)

        if tarfile.is_tarfile(tar_file_path):
            with tarfile.open(tar_file_path) as tar:
                tar.extractall(path=current_dir)
            print(f'{tar_file_path} 已解压缩至 {current_dir}')
        else:
            print(f'{tar_file_path} 不是一个有效的 tar 文件')
    else:
        print(f'{file_path} 已存在')
