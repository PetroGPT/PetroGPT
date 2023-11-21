import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from petrogpt import Evaluator


def cli_main():
    evaluator = Evaluator()
    evaluator.eval()


if __name__ == "__main__":
    cli_main()
