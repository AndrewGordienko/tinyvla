import runpy

from _bootstrap import bootstrap

bootstrap()

from tinyvla.task import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("tinyvla.task", run_name="__main__")
