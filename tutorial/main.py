import pickle
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## 1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients,
                                                                  cfg.batch_size)
    
    print(len(trainloaders), len(trainloaders[0].dataset))

    ## 3. Define your clients
    # This function returns a function which is able to instantiate a client of a particular id
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.model)

    ## 4. Define your strategy
    # strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
    #                                      min_fit_clients=cfg.num_clients_per_round_fit,   # How many clients are gonna be used per round for training
    #                                      fraction_evaluate=0.00001,
    #                                      min_evaluate_clients=cfg.num_clients_per_round_eval,  # How many clients are gonna be used for evaluation
    #                                      min_available_clients=cfg.num_clients,
    #                                      on_fit_config_fn=get_on_fit_config(cfg.config_fit),   
    #                                      evaluate_fn=get_evaluate_fn(cfg.num_classes,
    #                                                                  testloader)
    #                                      )
    
    strategy = instantiate(cfg.strategy, evaluate_fn=get_evaluate_fn(cfg.model,testloader))

    ## 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,      # To spawn a client
        num_clients=cfg.num_clients,    # To know the number of clients
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),     
        strategy=strategy,
        client_resources={'num_cpus': 2, 'num_gpus': 0},    # Optional argument
    )

    ## 6. Save results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history, "config": "some config"}

    with open(str(results_path), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()