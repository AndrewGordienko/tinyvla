import argparse

from _bootstrap import bootstrap

bootstrap()

from tinyvla.env import JOINT_NAMES, SO101Env, _run_render, _run_viewer

__all__ = ["JOINT_NAMES", "SO101Env"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="interactive viewer (use mjpython on macOS)")
    parser.add_argument("--render", action="store_true", help="save a still frame")
    args = parser.parse_args()
    if args.viewer:
        _run_viewer()
    else:
        _run_render()
