# -*- coding: utf-8 -*-
'''
This script serializes images and image-text pair annotations into LMDB files,
which supports more convenient dataset loading and random access to samples during training 
compared with TSV and Jsonl data files.

Now it also supports optional multiprocessing with a process pool.
'''

import argparse
import os
import json
import pickle
import subprocess
import mmap
from tqdm import tqdm
import lmdb
from multiprocessing import Pool, cpu_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="the directory which stores the image tsv files and the text jsonl annotations"
    )
    parser.add_argument(
        "--splits", type=str, required=True,
        help="specify the dataset splits which this script processes, concatenated by comma "
             "(e.g. train,valid,test)"
    )
    parser.add_argument(
        "--lmdb_dir", type=str, default=None,
        help="specify the directory which stores the output lmdb files. "
             "If set to None, the lmdb_dir will be set to {args.data_dir}/lmdb"
    )
    parser.add_argument(
        "--use_process_pool", action="store_true",
        help="whether to use a multiprocessing process pool for reading/processing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="number of worker processes in the process pool (effective only when --use_process_pool is set)"
    )
    return parser.parse_args()


# ---------------------- 工具函数：文件切片 & wc -l ---------------------- #

def get_total_lines(path: str) -> int:
    """Use `wc -l` (via subprocess) to get total line count, as你建议的那样."""
    cmd = f"cat '{path}' | wc -l"
    result = subprocess.run(
        cmd, shell=True, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out = result.stdout.strip()
    # wc 输出可能带空格，这里取第一个 token
    total = int(out.split()[0])
    return total


def compute_byte_chunks(path: str, num_workers: int):
    """
    将文件按字节划分为 num_workers 个不重叠的切片，并用 mmap 向后调整到换行符，以保证行对齐。
    返回 [(start_byte, end_byte), ...]，满足：
        0 = start_0 < end_0 = start_1 < end_1 = ... < end_{k-1} = file_size
    """
    file_size = os.path.getsize(path)
    if file_size == 0:
        return []

    if num_workers <= 1:
        return [(0, file_size)]

    # 避免 worker 数量 > 行数 / 文件字节数导致很多空切片
    num_workers = max(1, min(num_workers, file_size))

    chunk_size = file_size // num_workers
    chunks = []

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        start = 0
        for i in range(num_workers - 1):
            end = (i + 1) * chunk_size
            if end >= file_size:
                end = file_size
            else:
                # 将 end 向后移动到下一个 '\n' 的后一个字节，使新切片从行首开始
                nl_pos = mm.find(b"\n", end)
                if nl_pos == -1:
                    end = file_size
                else:
                    end = nl_pos + 1  # end 设为 '\n' 后一个字节位置

            if end <= start:
                # 极端情况（比如文件非常小），避免生成空区间
                end = file_size

            chunks.append((start, end))
            start = end
            if start >= file_size:
                break

        if start < file_size:
            chunks.append((start, file_size))

        mm.close()

    # 过滤掉空区间（理论上不会很多）
    chunks = [(s, e) for (s, e) in chunks if e > s]
    return chunks


def iter_lines_in_chunk(path: str, start: int, end: int):
    """
    给定 (start_byte, end_byte)，只读取该区间内的完整行。
    保证对每个切片内的行，顺序与原文件一致。
    """
    with open(path, "r", encoding="utf-8") as f:
        f.seek(start)
        current_pos = start

        while True:
            if current_pos >= end:
                break
            line = f.readline()
            if not line:
                break
            line_start = current_pos
            current_pos = f.tell()
            if line_start >= end:
                break
            yield line


# ---------------------- 子进程 worker 函数 ---------------------- #

def _process_pairs_chunk(args):
    """
    子进程 worker：处理 texts.jsonl 某一个字节切片内的所有行。
    返回值：List[pickled_dump]，其中每个 dump = pickle.dumps((image_id, text_id, text))
    """
    path, start, end = args
    results = []

    for raw_line in iter_lines_in_chunk(path, start, end):
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # 简单完整性检查（和原脚本一致）
        for field in ("text_id", "text", "image_ids"):
            assert field in obj, (
                f"Field {field} does not exist in line. "
                "Please check the integrity of the text annotation Jsonl file."
            )
        text_id = obj["text_id"]
        text = obj["text"]
        for image_id in obj["image_ids"]:
            dump = pickle.dumps((image_id, text_id, text))
            results.append(dump)

    return results


def _process_imgs_chunk(args):
    """
    子进程 worker：处理 imgs.tsv 某一个字节切片内的所有行。
    返回值：List[(image_id, b64_str)]
    """
    path, start, end = args
    results = []

    for raw_line in iter_lines_in_chunk(path, start, end):
        line = raw_line.strip()
        if not line:
            continue
        # 原脚本假定每行 "image_id\tb64"
        image_id, b64 = line.split("\t")
        results.append((image_id, b64))

    return results


# ---------------------- 处理 pairs（jsonl） ---------------------- #

def build_pairs_single_process(pairs_annotation_path, env_pairs):
    """
    完全沿用你原来的单进程 pairs 写法。
    """
    txn_pairs = env_pairs.begin(write=True)
    write_idx = 0
    with open(pairs_annotation_path, "r", encoding="utf-8") as fin_pairs:
        for line in tqdm(fin_pairs, desc=f"Serializing pairs (single): {os.path.basename(pairs_annotation_path)}"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for field in ("text_id", "text", "image_ids"):
                assert field in obj, (
                    f"Field {field} does not exist in line. "
                    "Please check the integrity of the text annotation Jsonl file."
                )
            for image_id in obj["image_ids"]:
                dump = pickle.dumps((image_id, obj["text_id"], obj["text"]))
                txn_pairs.put(key="{}".format(write_idx).encode("utf-8"), value=dump)
                write_idx += 1
                if write_idx % 5000 == 0:
                    txn_pairs.commit()
                    txn_pairs = env_pairs.begin(write=True)

    txn_pairs.put(key=b"num_samples", value="{}".format(write_idx).encode("utf-8"))
    txn_pairs.commit()
    env_pairs.close()
    print(f"Finished serializing {write_idx} pairs into {env_pairs.path().decode('utf-8') if hasattr(env_pairs, 'path') else 'pairs lmdb'}.")


def build_pairs_multiprocess(pairs_annotation_path, env_pairs, num_workers: int):
    """
    多进程版本：子进程负责解析 jsonl & 生成 dumps，主进程负责顺序写入 LMDB。
    """
    total_lines = get_total_lines(pairs_annotation_path)
    print(f"Total lines in {pairs_annotation_path}: {total_lines}")

    num_workers = max(1, min(num_workers, cpu_count()))
    print(f"Using {num_workers} worker processes for pairs.")

    chunks = compute_byte_chunks(pairs_annotation_path, num_workers)
    print(f"Computed {len(chunks)} byte-aligned chunks for pairs.")

    txn_pairs = env_pairs.begin(write=True)
    write_idx = 0

    # 用 imap 保证结果顺序与 chunks 顺序一致，避免一次性把所有结果装到内存里
    with Pool(processes=num_workers) as pool:
        chunk_args = [(pairs_annotation_path, s, e) for (s, e) in chunks]
        for chunk_result in tqdm(
            pool.imap(_process_pairs_chunk, chunk_args),
            total=len(chunk_args),
            desc=f"Serializing pairs (multi): {os.path.basename(pairs_annotation_path)}"
        ):
            for dump in chunk_result:
                txn_pairs.put(key="{}".format(write_idx).encode("utf-8"), value=dump)
                write_idx += 1
                if write_idx % 5000 == 0:
                    txn_pairs.commit()
                    txn_pairs = env_pairs.begin(write=True)

    txn_pairs.put(key=b"num_samples", value="{}".format(write_idx).encode("utf-8"))
    txn_pairs.commit()
    env_pairs.close()
    print(f"Finished serializing {write_idx} pairs into {env_pairs.path().decode('utf-8') if hasattr(env_pairs, 'path') else 'pairs lmdb'}.")


# ---------------------- 处理 imgs（tsv） ---------------------- #

def build_imgs_single_process(base64_path, env_img):
    """
    完全沿用你原来的单进程 imgs 写法。
    """
    txn_img = env_img.begin(write=True)
    write_idx = 0
    with open(base64_path, "r", encoding="utf-8") as fin_imgs:
        for line in tqdm(fin_imgs, desc=f"Serializing images (single): {os.path.basename(base64_path)}"):
            line = line.strip()
            if not line:
                continue
            image_id, b64 = line.split("\t")
            txn_img.put(key="{}".format(image_id).encode("utf-8"), value=b64.encode("utf-8"))
            write_idx += 1
            if write_idx % 1000 == 0:
                txn_img.commit()
                txn_img = env_img.begin(write=True)

    txn_img.put(key=b"num_images", value="{}".format(write_idx).encode("utf-8"))
    txn_img.commit()
    env_img.close()
    print(f"Finished serializing {write_idx} images into {env_img.path().decode('utf-8') if hasattr(env_img, 'path') else 'imgs lmdb'}.")


def build_imgs_multiprocess(base64_path, env_img, num_workers: int):
    """
    多进程版本：子进程负责解析 tsv 行，主进程负责顺序写入 LMDB。
    """
    total_lines = get_total_lines(base64_path)
    print(f"Total lines in {base64_path}: {total_lines}")

    num_workers = max(1, min(num_workers, cpu_count()))
    print(f"Using {num_workers} worker processes for images.")

    chunks = compute_byte_chunks(base64_path, num_workers)
    print(f"Computed {len(chunks)} byte-aligned chunks for images.")

    txn_img = env_img.begin(write=True)
    write_idx = 0

    with Pool(processes=num_workers) as pool:
        chunk_args = [(base64_path, s, e) for (s, e) in chunks]
        for chunk_result in tqdm(
            pool.imap(_process_imgs_chunk, chunk_args),
            total=len(chunk_args),
            desc=f"Serializing images (multi): {os.path.basename(base64_path)}"
        ):
            for image_id, b64 in chunk_result:
                txn_img.put(key="{}".format(image_id).encode("utf-8"), value=b64.encode("utf-8"))
                write_idx += 1
                if write_idx % 1000 == 0:
                    txn_img.commit()
                    txn_img = env_img.begin(write=True)

    txn_img.put(key=b"num_images", value="{}".format(write_idx).encode("utf-8"))
    txn_img.commit()
    env_img.close()
    print(f"Finished serializing {write_idx} images into {env_img.path().decode('utf-8') if hasattr(env_img, 'path') else 'imgs lmdb'}.")


# ---------------------- main ---------------------- #

if __name__ == "__main__":
    args = parse_args()
    assert os.path.isdir(args.data_dir), "The data_dir does not exist! Please check the input args..."

    # read specified dataset splits
    specified_splits = list(set(args.splits.strip().split(",")))
    print("Dataset splits to be processed: {}".format(", ".join(specified_splits)))

    # build LMDB data files
    if args.lmdb_dir is None:
        args.lmdb_dir = os.path.join(args.data_dir, "lmdb")

    for split in specified_splits:
        # open new LMDB files
        lmdb_split_dir = os.path.join(args.lmdb_dir, split)
        if os.path.isdir(lmdb_split_dir):
            print("We will overwrite an existing LMDB file {}".format(lmdb_split_dir))
        os.makedirs(lmdb_split_dir, exist_ok=True)

        # pairs LMDB
        lmdb_pairs = os.path.join(lmdb_split_dir, "pairs")
        env_pairs = lmdb.open(lmdb_pairs, map_size=1024 ** 4)

        pairs_annotation_path = os.path.join(args.data_dir, "{}_texts.jsonl".format(split))
        if args.use_process_pool:
            build_pairs_multiprocess(pairs_annotation_path, env_pairs, args.num_workers)
        else:
            build_pairs_single_process(pairs_annotation_path, env_pairs)

        # imgs LMDB
        lmdb_img = os.path.join(lmdb_split_dir, "imgs")
        env_img = lmdb.open(lmdb_img, map_size=1024 ** 4)

        base64_path = os.path.join(args.data_dir, "{}_imgs.tsv".format(split))
        if args.use_process_pool:
            build_imgs_multiprocess(base64_path, env_img, args.num_workers)
        else:
            build_imgs_single_process(base64_path, env_img)

    print("done!")

