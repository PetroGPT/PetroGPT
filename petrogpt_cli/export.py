import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from petrogpt import export_model


def cli_main():
    export_model()


if __name__ == "__main__":
    cli_main()
