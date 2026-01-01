import os
import base64
import json
from PIL import Image
from io import BytesIO


def convert_image_to_base64(image_path):
    """将图像文件转换为base64字符串"""
    img = Image.open(image_path)
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def process_txt_file(txt_file_path):
    """处理输入的txt文件，生成图像和文本数据"""
    image_data = []  # 存储图像的id和base64数据
    text_data = []  # 存储文本的id和图文匹配关系
    image_p_list = []

    image_id_counter = 1000000
    text_id_counter = 8000
    # 打开txt文件读取
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_path, caption = line.strip().split('\t')
            if image_path not in image_p_list:
                image_p_list.append(image_path)
                image_id_counter += 1
                base64_str = convert_image_to_base64(image_path)
                image_data.append({
                    'image_id': image_id_counter,
                    'base64': base64_str
                })

            text_data.append({
                'text_id': text_id_counter,
                'text': caption,
                'image_ids': [image_id_counter]
            })
            text_id_counter += 1

    return image_data, text_data


def save_image_data_to_tsv(image_data, tsv_file_path):
    """将图像数据保存为TSV文件"""
    with open(tsv_file_path, 'w', encoding='utf-8') as f:
        for image in image_data:
            f.write(f"{image['image_id']}\t{image['base64']}\n")


def save_text_data_to_jsonl(text_data, jsonl_file_path):
    """将文本数据保存为JSONL文件"""
    with open(jsonl_file_path, 'w', encoding='utf-8') as f:
        for text in text_data:
            json.dump(text, f, ensure_ascii=False)
            f.write('\n')


def main(txt_file_path, tsv_file_path, jsonl_file_path):
    image_data, text_data = process_txt_file(txt_file_path)
    save_image_data_to_tsv(image_data, tsv_file_path)
    save_text_data_to_jsonl(text_data, jsonl_file_path)
    print(f"数据处理完成！图像数据已保存到 {tsv_file_path}，文本数据已保存到 {jsonl_file_path}。")


if __name__ == '__main__':
    txt_file_path = '/home/lizongshu/projects/Chinese-CLIP/datasets/COCO-CN/coco_zh.txt'
    tsv_file_path = '/home/lizongshu/projects/Chinese-CLIP/datasets/COCO-CN/train_imgs.tsv'
    jsonl_file_path = '/home/lizongshu/projects/Chinese-CLIP/datasets/COCO-CN/train_texts.jsonl'
    main(txt_file_path, tsv_file_path, jsonl_file_path)
