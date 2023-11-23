# encoding=utf-8
'''
根据段落生成多轮问答
'''
import json
import argparse
from collections import namedtuple

from dalchemy.data import DalchemyData, TextHelper
from dalchemy.llms import ZhipuLLM
from dalchemy.parsers import TextBlocksParser

def convert_dict_to_namedtuple(d):
    return namedtuple("Struct", d.keys())(*d.values())


def main():
    parser = argparse.ArgumentParser(description="使用zhipu根据段落生成问题")
    parser.add_argument('--data_file', type=str, default='data/passages_all.json' ,help='问题')
    parser.add_argument('--out_file', type=str, default="petro_mqa.jsonl")
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end', type=int,default=-1)
    parser.add_argument('--api_key', required= True, type=str,help="请输入你的chatglm_pro的api_key")

    args = parser.parse_args()
    args = convert_dict_to_namedtuple(vars(args))
    helper = TextHelper()
    cfg = {
        "engine": "chatglm_pro",
        "api_key": args.api_key
    }
    llm = ZhipuLLM(**cfg)
    parser = TextBlocksParser(prefix_ls=["User", "Assistant"])

    def encode_prompt(text):
        prompt = "我现在要制作一批大模型训练的数据集，我希望你可以根据我输入的文本生成一组多轮对话，以达到我制作数据集的要求，我现在要输入的文本是： "
        prompt += text
        prompt += "请根据上述段生成问答，以User、Assistant开头："
        return prompt


    data_ls = helper.read_json(args.data_file)
    end = args.end
    if args.end == -1:
        end = len(data_ls)
    data_ls = data_ls[args.start: end]

    prompt_ls = [encode_prompt(data["content"]) for data in data_ls]
    batches = DalchemyData.make_batches(prompt_ls, batch_size=5)
    for batch in batches:
        results = llm.generate(batch, num_procs=len(batch))
        res = parser.parse_result(results)

        json_res = [json.dumps(r, ensure_ascii=False) for r in res]
        # save
        helper.write_lines_append(json_res, args.out_file, sep="\n")

if __name__ == '__main__':
    main()
