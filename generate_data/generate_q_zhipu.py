'''
根据段落生成问题。
'''
# encoding=utf-8
import sys
import json
import time
import re

from dalchemy.data import DalchemyData, TextHelper
from dalchemy.llms import AzureLLM, OpenAILLM,ZhipuLLM
from dalchemy.parsers import TextBlocksParser
from dalchemy.tasks import DalchemyTask
from tqdm import tqdm
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="使用zhipu根据段落生成问题")
    parser.add_argument('--passage_file', type=str, default='data/passages_all.json' ,help='问题')
    parser.add_argument('--outfile', type=str, default="data/query_ls_zhipu.json", help='输出JSON文件的路径，默认为 "data/query_ls_zhipu.json"')
    parser.add_argument('--size', type=int,default=5)
    parser.add_argument('--api_key', required= True, type=str,help="请输入你的chatglm_pro的api_key")

    args = parser.parse_args()

    helper = TextHelper()
    passage_file = args.passage_file
    outfile = args.outfile
    size = args.size
    passage_ls = helper.read_json(passage_file)[:size]
    prompt_q = "根据上述文本中与石油领域相关的内容与逻辑关系提出几个中文问题，注意，提出的问题应该提供充实的内容，使问题具有挑战性。"       
    cfg = {
    "engine": "chatglm_pro",
    "api_key": args.api_key
    }
    llm = ZhipuLLM(**cfg)

    res = []
    seen = set()
    qid = 0
    start = time.time()
    cost_secs = 0
    for passage in tqdm(passage_ls):
        pid = passage["pid"]
        content = passage["content"]
        prompt = f"{content}\n\n{prompt_q}"
        t1 = time.time()
        response = llm(prompt)
        t2 = time.time()
        dual = t2 - t1
        print("exec", dual)
        cost_secs += dual

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
    print("cost price", llm.cost_rmb, "\ncost sec", cost_secs)
    tokens = llm.cost_rmb * 1000 /0.01
    print("tokens/sec", tokens/cost_secs)

    helper.write_json(res, json_outfile=outfile)

if __name__ == '__main__' :
    main()
