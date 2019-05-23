from lib.utils.configurator import Configurator
import argparse
import importlib
import inflection

if __name__ == "__main__":
    # 1. Parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()

    # 2. Setup the experiment
    config = Configurator.configure(args.config_file)

    # 3. Unleash agent!
    agent_module = importlib.import_module(("lib.agents.{}").format(
        inflection.underscore(config["agent"])))
    Agent = getattr(agent_module, config["agent"])
    Agent(config).run()