'''
根据段落、问题、生成回答。

输入： 段落json、prompt、问题
输出： json
    字段： qid，query，answer，pid
'''


#encoding=utf-8
import sys
import json
import time

from dalchemy.data import DalchemyData, TextHelper
from dalchemy.llms import AzureLLM, OpenAILLM
from dalchemy.parsers import TextBlocksParser
from dalchemy.tasks import DalchemyTask
from tqdm import tqdm
import argparse

def main():
    helper = TextHelper()
    parser = argparse.ArgumentParser(description="使用zhipu根据段落生成问题")
    parser.add_argument('--data_file', type=str, default='data/query_ls.json' ,help='问题')
    parser.add_argument('--tmp_file', type=str, default="data/qa_ls.jsonl")
    parser.add_argument('--start', type=int,default=0)
    parser.add_argument('--end', type=int,default=-1)
    parser.add_argument('--api_key', required= True, type=str,help="请输入你的gpt的api_key")

    args = parser.parse_args()
    data_file = args.data_file
    tmp_file = args.tmp_file
    start = args.start 
    end = args.end

    prompt_q = "根据上述文本中与石油领域相关的内容与逻辑关系提出几个中文问题，注意，提出的问题应该提供充实的内容，使问题具有挑战性。"
    prompt_a_prefix = "请回答如下1个问题，注意生成的答案应该条理清晰，包含充实的内容，包括你自身的知识以及段落信息："

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
                prompt = AzureLLM.encode_chat_history(history)
                prompt_batch.append(prompt)
            yield prompt_batch


    # 4.调用llm，生成结果
    azure_cfg_liang = {
        # "engine": "gpt35-16k",
        "engine": "gpt35-16k",
        "api_type": "azure",
        "api_base": "https://cheneyoai2.openai.azure.com/",
        "api_version": "2023-03-15-preview",
        "api_key": args.api_key,
        # "max_tokens": 4096,
        "max_tokens": 8192,
    }

    llm = AzureLLM(**azure_cfg_liang)
    # print(llm("hello"))
    passage_id = start

    qid = 0
    for prompt_batch in generate_prompt_batches(passage2q_list, start=start, end=end):
        print("passage_id",passage_id)
        # print(prompt_batch[0])
        q_list = [AzureLLM.prompt_to_chatml(prompt)[-1] for prompt in prompt_batch]  # 不对，要转chatml模式
        q_list = [q["content"].replace(prompt_a_prefix, "") for q in q_list]

        passage_list = [AzureLLM.prompt_to_chatml(prompt)[0] for prompt in
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
        import random
        # time.sleep(random.randint(30, 45))
        passage_id += 1

    # 5.保存结果

if __name__ == '__main__':
    main()
