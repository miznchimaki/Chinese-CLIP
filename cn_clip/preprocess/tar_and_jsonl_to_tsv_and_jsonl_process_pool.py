import argparse
import os
import json
import tarfile
from PIL import Image
from io import BytesIO
import base64
import multiprocessing


def image_to_base64(image_path):
    img = Image.open(image_path)
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    base64_str = base64_str.decode("utf-8")  # str
    return base64_str


def read_jsonl(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data


def process_tar(tar_path, jsonl_data, tar_idx: int = 0, data_num_per_tar: int = 10000):
    image_data = []
    text_data = []
    image_id = tar_idx * data_num_per_tar
    text_id = 8 * tar_idx * data_num_per_tar

    with tarfile.open(tar_path, 'r') as tar:
        files = tar.getnames()
        image_files = [f for f in files if f.endswith('.jpg')]
        text_files = [f for f in files if f.endswith('.txt')]

        image_dict = {os.path.splitext(f)[0]: f for f in image_files}
        text_dict = {os.path.splitext(f)[0]: f for f in text_files}

        for entry in jsonl_data:
            key = entry['key']

            if key in image_dict:
                img_file = image_dict[key]
                img = tar.extractfile(img_file)
                img_path = os.path.join('/tmp', img_file)
                with open(img_path, 'wb') as f:
                    f.write(img.read())

                base64_img = image_to_base64(img_path)
                image_id += 1
                image_data.append((image_id, base64_img))

                txt_file = text_dict[key] if key in text_dict else None
                if txt_file:
                    txt = tar.extractfile(txt_file)
                    txt_content = txt.read().decode('utf-8').strip()
                    generated_caption = entry['generated_caption'].split('<|caption_sep|>')

                    for caption in generated_caption:
                        if caption:
                            text_id += 1
                            text_data.append({
                                "text_id": text_id,
                                "text": caption.strip(),
                                "image_ids": [image_id]
                            })

    return image_data, text_data


def worker(jsonl_p, tar_dir, data_num_per_tar, idx):
    tar_name = os.path.basename(jsonl_p).replace('.jsonl', '.tar')
    tar_p = os.path.join(tar_dir, tar_name)
    if not os.path.exists(tar_p):
        raise FileNotFoundError(f'corresponding raw tar path - {tar_p}, does not exist!')

    jsonl_data = read_jsonl(jsonl_p)
    img_data, txt_data = process_tar(
        tar_p,
        jsonl_data,
        tar_idx=idx,
        data_num_per_tar=data_num_per_tar
    )
    return img_data, txt_data


def main(jsonl_files_list, tar_dir, jsonl_dir, pool_size):
    all_image_data = []
    all_text_data = []
    data_num_per_tar = 10000

    # Initialize the process pool
    with multiprocessing.Pool(processes=pool_size) as pool:
        results = pool.starmap(worker, [(jsonl_p, tar_dir, data_num_per_tar, idx) for idx, jsonl_p in enumerate(jsonl_files_list)])

    # Collecting results from all workers
    for img_data, txt_data in results:
        all_image_data.extend(img_data)
        all_text_data.extend(txt_data)

    with open(os.path.join(jsonl_dir, 'train_imgs.tsv'), 'w', encoding='utf-8') as img_file:
        for img_id, base64_str in all_image_data:
            img_file.write(f"{img_id}\t{base64_str}\n")

    with open(os.path.join(jsonl_dir, 'train_texts.jsonl'), 'w', encoding='utf-8') as text_file:
        for text in all_text_data:
            text_file.write(json.dumps(text, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('directory arguments parser for transforming tar & jsonl raw data into tsv & jsonl needed data')
    parser.add_argument('--tar-dir', default=None, type=str, help='raw tar files data directory')
    parser.add_argument('--jsonl-dir', default=None, type=str, help='raw jsonl files data directory')
    parser.add_argument('--pool-size', default=None, type=int, help='Size of the process pool')
    args = parser.parse_args()

    jsonl_files_p = [os.path.join(args.jsonl_dir, jsonl_name) for jsonl_name in os.listdir(args.jsonl_dir) if jsonl_name.endswith('.jsonl')]
    # Adjust pool size based on the number of jsonl files
    pool_size = min(args.pool_size, len(jsonl_files_p)) if args.pool_size else len(jsonl_files_p)

    main(jsonl_files_p, args.tar_dir, args.jsonl_dir, pool_size)
