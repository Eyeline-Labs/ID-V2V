import torch
import argparse
import importlib
from typing import Dict, Any

def load_parser(module_name: str) -> argparse.ArgumentParser:
    module = importlib.import_module(module_name)
    if not hasattr(module, "get_parser"):
        raise ValueError(f"{module_name} undefined get_parser()")
    return module.get_parser()

def filter_args(args: Dict[str, Any], parser: argparse.ArgumentParser) -> Dict[str, Any]:
    known_args = set()
    for action in parser._actions:
        if action.dest and action.dest != "help":
            known_args.add(action.dest)
    return {k: v for k, v in args.items() if k in known_args}

def main():

    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("--base", type=str, default='ltx', choices=['ltx', 'wan'])
    pipeline_args, _ = main_parser.parse_known_args()

    if pipeline_args.base in ["ltx"]:
        preproccess_name, inference_name = "vace_preproccess", "vace_ltx_inference"
    else:
        preproccess_name, inference_name = "vace_preproccess", "vace_wan_inference"

    preprocess_parser = load_parser(preproccess_name)
    inference_parser = load_parser(inference_name)

    for parser in [preprocess_parser, inference_parser]:
        for action in parser._actions:
            if action.dest != "help":
                main_parser._add_action(action)

    cli_args = main_parser.parse_args()
    args_dict = vars(cli_args)

    # run preprocess
    preprocess_args = filter_args(args_dict, preprocess_parser)
    preprocesser = importlib.import_module(preproccess_name)
    preprocess_output = preprocesser.main(preprocess_args)
    print("preprocess_output:", preprocess_output)

    del preprocesser
    torch.cuda.empty_cache()

    # run inference
    inference_args = filter_args(args_dict, inference_parser)
    inference_args.update(preprocess_output)
    preprocess_output = importlib.import_module(inference_name).main(inference_args)
    print("preprocess_output:", preprocess_output)


if __name__ == "__main__":
    main()