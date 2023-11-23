'''
根据段落、问题、生成回答。
'''


#encoding=utf-8
import os
import sys
import json
import time
import random

from dalchemy.data import DalchemyData, TextHelper
from dalchemy.llms import AzureLLM, OpenAILLM,ZhipuLLM,DalchemyLLM
from dalchemy.parsers import TextBlocksParser
from dalchemy.tasks import DalchemyTask
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="使用zhipu根据段落、问题，生成回答")
    parser.add_argument('--query_dir', type=str, default='query_ls2')
    parser.add_argument('--answer_dir', type=str, default="answer_ls2")
    parser.add_argument('--part', required=True,type=int)
    parser.add_argument('--start', type=int,default=0,help="根据输出文件的段落数量，来决定start多少")
    parser.add_argument('--end', type=int,default=-1, help="总共4w 这个是根据段落来划分的")
    parser.add_argument('--key', required= True, dest='key_ls', action='append', help='请输入你的key')

    args = parser.parse_args()
    helper = TextHelper()
    part = args.part
    query_dir = args.query_dir
    answer_dir = args.answer_dir
    data_file = f'data/{query_dir}/part{part}.json'
    prompt_q = "根据上述文本中与石油领域相关的内容与逻辑关系提出几个中文问题，注意，提出的问题应该提供充实的内容，使问题具有挑战性。"
    prompt_a_prefix = "请回答如下1个问题，注意生成的答案应该条理清晰，包含充实的内容，包括你自身的知识以及段落信息："
    tmp_file = f"data/{answer_dir}/qa_ls_zhipu{part}.jsonl"

    outdir= os.path.dirname(tmp_file)
    if not os.path.exists(outdir): os.makedirs(outdir)

    start = args.start 
    end = args.end
    stime = 7

    if not os.path.exists(tmp_file):
        start = 0
    else:
        qa_ls = helper.read_lines(tmp_file)
        qa_ls = [line for line in qa_ls if line.strip()!=""]
        print("already len qa", len(qa_ls))
        if len(qa_ls)>0:
            start = len(qa_ls)
            print("start", start)
    data_ls = helper.read_json(data_file)
    passage2q_list = {}

    # 1.构建段落到问题的字典
    for data in tqdm(data_ls):
        question = data["query"].strip()
        passage = str(data["pid"])
        if question not in passage2q_list.get(passage, []):  # 防止重复
            passage2q_list[passage] = passage2q_list.get(passage, []) + [question]


    # 2.构造history

    def get_history(passage, passage2q_list):
        ''' 根据段落，和段落到问题列表的字典,生成history'''
        history = []
        prompt = f"{passage}\n\n{prompt_q}"
        history.append({"role": "user", "content": prompt})
        q_ls = passage2q_list[passage]
        # q_ls = [f"问题{i}：{q}" for i, q in enumerate(q_ls, start=1)]
        q_ls = [f"{i}. {q}" for i, q in enumerate(q_ls, start=1)]
        response = "\n".join(q_ls)
        history.append({"role": "assistant", "content": response})
        return history
    
    # 3.迭代所有question，生成prompt
    print(len(passage2q_list.items()))
    # for k,v in passage2q_list.items():
    #     print(len(v))


    def generate_prompt_batches(passage2q_list, start=0, end=-1):
        print(len(passage2q_list),"passage2q_list")
        if end == -1:
            end = len(passage2q_list.items())
        for pid, (passage, q_list) in enumerate(passage2q_list.items()):
            prompt_batch = []  # 一个段落的所有question作为prompt_batch
            if pid < start: continue
            if pid >= end: break
            for i, q in enumerate(q_list, start=1):
                history = get_history(passage, passage2q_list)
                query = f"{prompt_a_prefix} {q}"
                history.append({"role": "user", "content": query})
                prompt = DalchemyLLM.encode_chat_history(history)
                prompt_batch.append(prompt)
            yield prompt_batch

    key_ls = args.key_ls

    api_key = key_ls[part]
    cfg = {
        "engine": "chatglm_pro",
        "api_key": api_key
    }
    llm = ZhipuLLM(**cfg)
    # print(llm("hello"))
    passage_id = start

    qid = 0
    for prompt_batch in generate_prompt_batches(passage2q_list, start=start, end=end):
        print("passage_id",passage_id)
        # print(prompt_batch[0])
        q_list = [DalchemyLLM.prompt_to_chatml(prompt)[-1] for prompt in prompt_batch]  # 不对，要转chatml模式
        q_list = [q["content"].replace(prompt_a_prefix, "") for q in q_list]

        passage_list = [DalchemyLLM.prompt_to_chatml(prompt)[0] for prompt in
                        prompt_batch]  # user(passage) response(q_list) user(q)
        passage_list = [p["content"].replace(prompt_q, "").strip() for p in passage_list]

        num_procs = len(prompt_batch)
        step_results = llm.generate(prompt_batch, is_chat=True, num_procs=num_procs)
        # step_results = [str(idx) for idx in range(len(prompt_batch))]
        res = []
        for q, a, passage in zip(q_list, step_results, passage_list):
            record = {"qid": str(qid),"query": q.strip(), "answer": a.strip(), "pid": str(passage)}  # 获取context
            res.append(json.dumps(record, ensure_ascii=False))
            qid += 1

        # 调用llm，使用chat模式，生成结果，并保存
        helper.write_lines_append(res, tmp_file, sep="\n")
        # time.sleep(random.randint(30, 45))
        bsz = len(prompt_batch)
        time.sleep(bsz * 2.5)
        # time.sleep(random.randint(stime,stime+3))
        #time.sleep(random.randint(5, 10))
        passage_id += 1


    print("total cost rmb:", llm.cost_rmb)

if __name__ == '__main__' :
    main()

