import json
import re
import zhconv
import argparse

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]
    
def text_filter(text):
    keywords = [
        r'如[图表].*?所示',
        r'[表图][\d]*-[\d]*',
        r'公式\([\d\.-]*\)',
        r'例[\d]*-[\d]*',
        r'背景',
        r'如表\s*\d*\.*\d*',
        r'如图\s*\d*\.*\d*',
        r'如表\s*[0-9a-zA-Z]*',
        r'如图\s*[0-9a-zA-Z]*',
    ]
    pattern = re.compile('|'.join(keywords), re.IGNORECASE)
    filtered_text = pattern.sub('', text)
    return filtered_text

def convert_to_simplified(text):
    return zhconv.convert(text, 'zh-cn')

def process_dialogues(dialogues):
    processed_dialogues = []
    pattern = re.compile(r"这段文本.*?[？\?]")

    for dialogue in dialogues:
        new_dialogue = []
        for turn in dialogue:
            if pattern.search(turn["User"]):
                continue  # 只跳过包含特定问题的那一次对话
            turn["User"] = turn["User"].replace("好的，让我们开始吧。", "")
            turn["User"] = turn["User"].replace("好的，让我们开始。", "")
            turn["User"] = turn["User"].replace("根据本文内容，", "")
            turn["User"] = turn["User"].replace("我注意到文本中提到了", "")
            turn["User"] = turn["User"].replace("文中提到的", "")
            turn["User"] = turn["User"].replace("文中提到了，", "")
            turn["User"] = turn["User"].replace("这本书提出的", "")
            turn["User"] = turn["User"].replace("这本书是", "")
            turn["User"] = turn["User"].replace("这本书", "")
            turn["User"] = turn["User"].replace("根据文本内容，", "")
            turn["User"] = text_filter(convert_to_simplified(turn["User"]))
            turn["Assistant"] = text_filter(convert_to_simplified(turn["Assistant"]))
            new_dialogue.append(turn)

        if new_dialogue:  # 检查处理后的对话是否为空
            processed_dialogues.append(new_dialogue)

    return processed_dialogues

def main():
    parser = argparse.ArgumentParser(description="处理成最终的多轮对话数据集")
    parser.add_argument('--data_file', type=str, default='petro_mqa.jsonl' ,help='输入的数据集路径')
    parser.add_argument('--outfile', type=str, default="processed_petro_mqa.jsonl", help='输出JSON文件的路径，默认为 "processed_petro_mqa.jsonl"')

    args = parser.parse_args()
    # 加载并处理 JSONL 文件
    file_path = args.data_file
    dialogues = load_jsonl(file_path)
    processed_dialogues = process_dialogues(dialogues)

    # 将处理后的数据保存回文件
    with open(args.outfile, 'w', encoding='utf-8') as outfile:
        for dialogue in processed_dialogues:
            json.dump(dialogue, outfile, ensure_ascii=False)
            outfile.write('\n')

if __name__ == '__main__' :
    main()