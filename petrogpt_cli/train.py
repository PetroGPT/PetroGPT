import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from petrogpt import run_exp


def cli_main():
    run_exp()


if __name__ == "__main__":
    cli_main()
