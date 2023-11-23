from glob import glob
import os
import re
from docx import Document as DP
import zhconv

def has_non_alphanumeric(text, threshold=0.5):
    # 移除空格和标点符号
    cleaned_text = re.sub(r'\s+|[^\w\s]', '', text)
    if cleaned_text.strip() == "": return False

    # 计算数字和英文字母的比例
    total_chars = len(cleaned_text)
    total_digits = len(re.findall(r'\d', cleaned_text))
    total_letters = len(re.findall(r'[a-zA-Z]', cleaned_text))
    ratio = (total_digits + total_letters) / total_chars

    # 判断比例是否超过阈值
    if ratio > threshold:
        return False
    else:
        return True

def read_pydocx(file_path):
    res = []
    docx_file = DP(file_path)
    for paragraph in docx_file.paragraphs:
        if not paragraph.text.strip():
            continue
        text = zhconv.convert(paragraph.text, 'zh-cn')
        res.append(text)
    return res

def get_doc_name(file_path):
    doc_name = os.path.basename(file_path)

    doc_name_regex = re.compile(r'^(?P<doc>.*?)_[\d]+【.*.docx$')
    m_d = doc_name_regex.match(doc_name)
    if m_d:
        doc_name = m_d["doc"]
    doc_name = doc_name.replace(".docx", "")
    return doc_name

# files = glob(pathname="./petro/*/*/*.docx")
files = glob(pathname="D:/datasets/petro/石油数据集/*/*/*.docx")
print(len(files))
res = []
for file in files:
    doc_name = get_doc_name(file)
    if doc_name.startswith("~$"): 
        continue
    content_ls = read_pydocx(file)
    text = "\n".join(content_ls)
    text = f"标题: {doc_name}" + text + "\n\n"
    if not has_non_alphanumeric(text):
        print(text[:50])
    res.append(text)
from dalchemy.data import TextHelper
h = TextHelper()
h.write_lines(res, outfile="petro_pretrain_all.txt")