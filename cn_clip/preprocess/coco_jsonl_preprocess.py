import os
import json
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def process_images_and_texts(image_dir, raw_jsonl_p):
    # 输出文件路径
    tsv_output = '/home/lizongshu/projects/Chinese-CLIP/datasets/COCO-CN-2/train_imgs.tsv'
    jsonl_output = '/home/lizongshu/projects/Chinese-CLIP/datasets/COCO-CN-2/train_texts.jsonl'
    img_name_list = []

    # 打开输出文件
    with open(tsv_output, 'w', encoding='utf-8') as tsv_file, open(jsonl_output, 'w', encoding='utf-8') as jsonl_file:
        with open(raw_jsonl_p, 'r', encoding='utf-8') as f:
            caption_data = f.readlines()
        text_id_map = {}

        for line in tqdm(caption_data, desc='annotations read & image write'):
            data = json.loads(line.strip())
            image_id = data['image_id']
            image_name = str(data['image_id']).zfill(12) + '.jpg'
            caption_zh = data['caption_zh'].strip()
            caption_id = int(data['id'])

            image_path = os.path.join(image_dir, image_name)
            if image_name not in img_name_list:
                img_name_list.append(image_name)
                img = Image.open(image_path)
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data).decode("utf-8")
                tsv_file.write(f"{image_id}\t{base64_str}\n")

            # 准备图文匹配关系
            if caption_id not in text_id_map:
                text_id_map[caption_id] = {'text': caption_zh, 'image_ids': []}
            text_id_map[caption_id]['image_ids'].append(image_id)

        # 写入train_texts.jsonl
        for text_id, text_info in tqdm(text_id_map.items(), desc='captions write'):
            jsonl_file.write(json.dumps({"text_id": text_id, "text": text_info['text'], "image_ids": text_info['image_ids']}, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    image_dir = '/home/lizongshu/datasets/coco/images/train2017'
    jsonl_file = '/home/lizongshu/projects/Chinese-CLIP/datasets/COCO-CN-2/coco_2017_cap_zh.jsonl'
    process_images_and_texts(image_dir, jsonl_file)
