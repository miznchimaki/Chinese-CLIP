# -*- coding: utf-8 -*-
'''
This script serializes images and image-text pair annotations into LMDB files,
which supports more convenient dataset loading and random access to samples during training 
compared with TSV and Jsonl data files.

Now it also supports optional multiprocessing with a process pool,
where data chunks are split by line ranges (按行切分).
'''

import argparse
import os
import json
import pickle
import subprocess
from tqdm import tqdm
import lmdb
from multiprocessing import Pool


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


# ---------------------- 工具函数：行数统计 & 行切分 ---------------------- #

def get_total_lines(path: str) -> int:
    """
    Use `cat path | wc -l` (via subprocess) to get total line count,
    按你最初的建议实现。
    """
    cmd = f"cat '{path}' | wc -l"
    result = subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out = result.stdout.strip()
    total = int(out.split()[0])
    return total


def get_total_lines_fast(path, block_size=8 * 1024 * 1024):
    size = os.path.getsize(path)
    if size == 0:
        return 0

    count = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            count += block.count(b"\n")

        # 检查最后一个字节是不是 '\n'，如果不是，说明最后一行没有换行符
        f.seek(-1, os.SEEK_END)
        last_byte = f.read(1)
        if last_byte != b"\n":
            count += 1

    return count


def compute_line_chunks(total_lines: int, num_workers: int):
    """
    根据总行数和进程数，按行号均匀切分成若干区间。
    返回 [(start_line, end_line), ...]，满足：
        0 = start_0 < end_0 = start_1 < ... < end_{k-1} = total_lines
    """
    if total_lines == 0:
        return []

    num_workers = max(1, min(num_workers, total_lines))

    base = total_lines // num_workers
    rem = total_lines % num_workers

    chunks = []
    start = 0
    for i in range(num_workers):
        # 前 rem 个 worker 每人多 1 行
        length = base + (1 if i < rem else 0)
        if length <= 0:
            break
        end = start + length
        chunks.append((start, end))
        start = end

    # 理论上最后一个 end == total_lines
    return chunks


def iter_lines_by_index(path: str, start_line: int, end_line: int):
    """
    按行号区间 [start_line, end_line) 迭代文件中的行。
    注意：这是最直接的实现，每个 worker 都是从头 enumerate，
    然后跳过前 start_line 行。
    """
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_line:
                continue
            if idx >= end_line:
                break
            yield line


# ---------------------- 子进程 worker 函数（按行切分） ---------------------- #

def _process_pairs_chunk_by_line(args):
    """
    子进程 worker：处理 texts.jsonl 的某一行区间 [start_line, end_line)。
    返回：List[pickled_dump]，dump = pickle.dumps((image_id, text_id, text))
    """
    path, start_line, end_line = args
    results = []

    for raw_line in iter_lines_by_index(path, start_line, end_line):
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
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


def _process_imgs_chunk_by_line(args):
    """
    子进程 worker：处理 imgs.tsv 的某一行区间 [start_line, end_line)。
    返回：List[(image_id, b64)]
    """
    path, start_line, end_line = args
    results = []

    for raw_line in iter_lines_by_index(path, start_line, end_line):
        line = raw_line.strip()
        if not line:
            continue
        image_id, b64 = line.split("\t")
        results.append((image_id, b64))

    return results


# ---------------------- 单进程版本（和你原来逻辑等价） ---------------------- #

def build_pairs_single_process(pairs_annotation_path: str, env_pairs):
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
                txn_pairs.put(key=str(write_idx).encode("utf-8"), value=dump)
                write_idx += 1
                if write_idx % 5000 == 0:
                    txn_pairs.commit()
                    txn_pairs = env_pairs.begin(write=True)

    txn_pairs.put(key=b"num_samples", value=str(write_idx).encode("utf-8"))
    txn_pairs.commit()
    env_pairs.close()

    print(f"Finished serializing {write_idx} pairs into {env_pairs}.")


def build_imgs_single_process(base64_path: str, env_img):
    txn_img = env_img.begin(write=True)
    write_idx = 0

    with open(base64_path, "r", encoding="utf-8") as fin_imgs:
        for line in tqdm(fin_imgs, desc=f"Serializing images (single): {os.path.basename(base64_path)}"):
            line = line.strip()
            if not line:
                continue
            image_id, b64 = line.split("\t")
            txn_img.put(key=str(image_id).encode("utf-8"), value=b64.encode("utf-8"))
            write_idx += 1
            if write_idx % 1000 == 0:
                txn_img.commit()
                txn_img = env_img.begin(write=True)

    txn_img.put(key=b"num_images", value=str(write_idx).encode("utf-8"))
    txn_img.commit()
    env_img.close()

    print(f"Finished serializing {write_idx} images into {env_img}.")


# ---------------------- 多进程版本（按行切分） ---------------------- #

def build_pairs_multiprocess_line_based(pairs_annotation_path: str, env_pairs, num_workers: int):
    total_lines = get_total_lines(pairs_annotation_path)
    print(f"Total lines in {pairs_annotation_path}: {total_lines}")

    if total_lines == 0:
        txn_pairs = env_pairs.begin(write=True)
        txn_pairs.put(key=b"num_samples", value=b"0")
        txn_pairs.commit()
        env_pairs.close()
        print(f"No pairs to serialize for {pairs_annotation_path}.")
        return

    num_workers = max(1, num_workers)
    print(f"Using {num_workers} worker processes for pairs (line-based).")

    chunks = compute_line_chunks(total_lines, num_workers)
    print(f"Computed {len(chunks)} line-based chunks for pairs.")

    txn_pairs = env_pairs.begin(write=True)
    write_idx = 0

    # 保证顺序：按照 chunks 顺序提交任务，用 imap 顺序收结果
    with Pool(processes=num_workers) as pool:
        chunk_args = [(pairs_annotation_path, s, e) for (s, e) in chunks]
        for chunk_result in tqdm(
            pool.imap(_process_pairs_chunk_by_line, chunk_args),
            total=len(chunk_args),
            desc=f"Serializing pairs (multi, line-based): {os.path.basename(pairs_annotation_path)}"
        ):
            for dump in chunk_result:
                txn_pairs.put(key=str(write_idx).encode("utf-8"), value=dump)
                write_idx += 1
                if write_idx % 5000 == 0:
                    txn_pairs.commit()
                    txn_pairs = env_pairs.begin(write=True)

    txn_pairs.put(key=b"num_samples", value=str(write_idx).encode("utf-8"))
    txn_pairs.commit()
    env_pairs.close()

    print(f"Finished serializing {write_idx} pairs into {env_pairs}.")
    return total_lines


def build_imgs_multiprocess_line_based(base64_path: str, env_img, num_workers: int, total_lines: int):
    # total_lines = get_total_lines(base64_path)
    # total_lines = get_total_lines_fast(base64_path)
    print(f"Total lines in {base64_path}: {total_lines}")

    if total_lines == 0:
        txn_img = env_img.begin(write=True)
        txn_img.put(key=b"num_images", value=b"0")
        txn_img.commit()
        env_img.close()
        print(f"No images to serialize for {base64_path}.")
        return

    num_workers = max(1, num_workers)
    print(f"Using {num_workers} worker processes for images (line-based).")

    chunks = compute_line_chunks(total_lines, num_workers)
    print(f"Computed {len(chunks)} line-based chunks for images.")

    txn_img = env_img.begin(write=True)
    write_idx = 0

    with Pool(processes=num_workers) as pool:
        chunk_args = [(base64_path, s, e) for (s, e) in chunks]
        for chunk_result in tqdm(
            pool.imap(_process_imgs_chunk_by_line, chunk_args),
            total=len(chunk_args),
            desc=f"Serializing images (multi, line-based): {os.path.basename(base64_path)}"
        ):
            for image_id, b64 in chunk_result:
                txn_img.put(key=str(image_id).encode("utf-8"), value=b64.encode("utf-8"))
                write_idx += 1
                if write_idx % 1000 == 0:
                    txn_img.commit()
                    txn_img = env_img.begin(write=True)

    txn_img.put(key=b"num_images", value=str(write_idx).encode("utf-8"))
    txn_img.commit()
    env_img.close()

    print(f"Finished serializing {write_idx} images into {env_img}.")


# ---------------------- main ---------------------- #

if __name__ == "__main__":
    args = parse_args()
    assert os.path.isdir(args.data_dir), "The data_dir does not exist! Please check the input args..."

    specified_splits = list(set(args.splits.strip().split(",")))
    print("Dataset splits to be processed: {}".format(", ".join(specified_splits)))

    if args.lmdb_dir is None:
        args.lmdb_dir = os.path.join(args.data_dir, "lmdb")

    use_pool = args.use_process_pool
    num_workers = args.num_workers

    for split in specified_splits:
        # LMDB 目录
        lmdb_split_dir = os.path.join(args.lmdb_dir, split)
        if os.path.isdir(lmdb_split_dir):
            print("We will overwrite an existing LMDB file {}".format(lmdb_split_dir))
        os.makedirs(lmdb_split_dir, exist_ok=True)

        # ---------- 处理 pairs (jsonl) ----------
        lmdb_pairs = os.path.join(lmdb_split_dir, "pairs")
        env_pairs = lmdb.open(lmdb_pairs, map_size=1024 ** 4)
        pairs_annotation_path = os.path.join(args.data_dir, "{}_texts.jsonl".format(split))

        if use_pool:
            total_lines = build_pairs_multiprocess_line_based(pairs_annotation_path, env_pairs, num_workers)
        else:
            build_pairs_single_process(pairs_annotation_path, env_pairs)

        # ---------- 处理 imgs (tsv) ----------
        lmdb_img = os.path.join(lmdb_split_dir, "imgs")
        env_img = lmdb.open(lmdb_img, map_size=1024 ** 4)
        base64_path = os.path.join(args.data_dir, "{}_imgs.tsv".format(split))

        if use_pool:
            build_imgs_multiprocess_line_based(base64_path, env_img, num_workers, total_lines)
        else:
            build_imgs_single_process(base64_path, env_img)

    print("done!")

