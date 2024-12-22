
from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl

from hydra.utils import instantiate

from model import train, test

class FLowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 model_cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = instantiate(model_cfg)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # Receives from the server the parameters of the global model and a set of instructions(like the hyperparameters)
    def fit(self, paramaters, config):
        # Copy parameters sent by the server into local model
        self.set_parameters(paramaters)

        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Do local training
        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}
    
    # Receives the parameters of the global model and evaluates the using the local validation set and returns the corresponding metrics
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Copy parameters sent by the server into local model
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"accuracy": accuracy}
    

# This function will be called by the main.py
# This is a function for the server to spawn clients
def generate_client_fn(trainloaders, valloaders, model_cfg):

    def client_fn(cid: str):

        return FLowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            model_cfg=model_cfg,
                            )

    return client_fn