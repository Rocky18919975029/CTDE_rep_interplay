
"""Train an algorithm."""

import argparse
import json
import sys
from pathlib import Path

# A server environment may still contain an editable install of an older HARL
# checkout.  Running this source entry point must always use its sibling `harl`
# package, independent of site-packages/editable-finder ordering.
REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_string = str(REPO_ROOT)
if sys.path[0] != repo_root_string:
    sys.path.insert(0, repo_root_string)

import harl

HARL_SOURCE = Path(harl.__file__).resolve()
if REPO_ROOT not in HARL_SOURCE.parents:
    raise RuntimeError(
        f"Imported HARL from {HARL_SOURCE}, expected the checkout at {REPO_ROOT}"
    )


def main():
    """Main function."""
    print(f"HARL source: {HARL_SOURCE}")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    from harl.utils.configs_tools import get_defaults_yaml_args, update_args

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]] 
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)
    requested_exp_name = args["exp_name"]
    if args["load_config"] != "": 
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
       
        args["exp_name"] = (
            requested_exp_name
            if requested_exp_name is not None
            else all_config["main_args"]["exp_name"]
        )
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  
        args["exp_name"] = requested_exp_name or "installtest"
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  

    if args["env"] == "dexhands":
        import isaacgym  

    
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

   
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
