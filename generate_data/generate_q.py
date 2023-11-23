'''
根据段落生成问题。

输入： 段落json、prompt
输出： json
    字段： qid，query，pid
'''

# encoding=utf-8
import sys
import json
import time
import re

from dalchemy.data import DalchemyData, TextHelper
from dalchemy.llms import AzureLLM, OpenAILLM
from dalchemy.parsers import TextBlocksParser
from dalchemy.tasks import DalchemyTask
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="使用zhipu根据段落生成问题")
    parser.add_argument('--passage_file', type=str, default='data/passages_all.json' ,help='问题')
    parser.add_argument('--outfile', type=str, default="data/query_ls.json", help='输出JSON文件的路径，默认为 "data/query_ls.json"')
    parser.add_argument('--size', type=int,default=10)
    parser.add_argument('--api_key', required= True, type=str,help="请输入你的chatglm_pro的api_key")

    args = parser.parse_args()

    helper = TextHelper()
    passage_file = args.passage_file
    outfile = args.outfile
    size = args.size
    passage_ls = helper.read_json(passage_file)[:size]
    prompt_q = "根据上述文本中与石油领域相关的内容与逻辑关系提出几个中文问题，注意，提出的问题应该提供充实的内容，使问题具有挑战性。"       

    azure_cfg_liang = {
    # "engine": "gpt35-16k",
    "engine": "gpt35-16k",
    "api_type": "azure",
    "api_base": "https://cheneyoai2.openai.azure.com/",
    "api_version": "2023-03-15-preview",
    "api_key": args.api_key,
    "max_tokens": 4096,
    # "max_tokens": 8192,
    }
    llm = AzureLLM(**azure_cfg_liang)

    res = []
    seen = set()
    qid = 0
    for passage in passage_ls:
        pid = passage["pid"]
        content = passage["content"]
        prompt = f"{content}\n\n{prompt_q}"

        response = llm(prompt)
        query_ls = response.split("\n")
        pattern = r'^\d+\.\s*(.*)$'
        for query in query_ls:
            query = re.sub(pattern, r'\1', query)
            query = query.strip()
            if query not in seen:
                seen.add(query)
            else:
                continue
            record = {"qid": str(qid), "query": query, "pid": pid}
            res.append(record)
            qid += 1

    helper.write_json(res, json_outfile=outfile)

if __name__ == '__main__':
    main()
