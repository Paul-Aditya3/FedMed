import torch
import copy
from collections import OrderedDict

class Server:
    def __init__(self, conf, global_model, num_clients):
        """
        Initialize FedMix server
        
        Args:
            conf: Configuration dictionary
            global_model: Global neural network model
            num_clients: Number of participating clients
        """
        self.conf = conf
        self.global_model = global_model
        self.num_clients = num_clients
        
    def aggregate(self, client_updates, weights=None):
        """
        Aggregate client updates using FedAvg
        
        Args:
            client_updates: List of client model state dictionaries
            weights: Optional list of aggregation weights
        """
        if weights is None:
            weights = [1.0 / self.num_clients] * self.num_clients
            
        aggregated_dict = OrderedDict()
        
        # Initialize with first client's updates
        for key in client_updates[0].keys():
            aggregated_dict[key] = torch.zeros_like(client_updates[0][key])
            
        # Weighted average of all clients
        for client_idx, client_dict in enumerate(client_updates):
            for key in aggregated_dict.keys():
                aggregated_dict[key] += client_dict[key] * weights[client_idx]
                
        self.global_model.load_state_dict(aggregated_dict)
        return copy.deepcopy(self.global_model)

def train_fedmix(conf, global_model, clients, test_loader=None):
    """
    Main FedMix training loop
    
    Args:
        conf: Configuration dictionary
        global_model: Initial global model
        clients: List of Client objects
        test_loader: Optional test dataloader for evaluation
    """
    server = Server(conf, global_model, len(clients))
    
    # Calculate mean data for all clients
    mean_batch = conf.get("mean_batch_size", 32)
    all_Xmean, all_ymean = [], []
    for client in clients:
        Xmean, ymean = client.calculate_mean_data(mean_batch)
        all_Xmean.append(Xmean)
        all_ymean.append(ymean)
    
    # Concatenate all mean data
    global_Xmean = torch.cat(all_Xmean, dim=0)
    global_ymean = torch.cat(all_ymean, dim=0)
    
    # Share global mean data with all clients
    for client in clients:
        client.get_mean_data(global_Xmean, global_ymean)
    
    # Federation rounds
    for round_idx in range(conf["num_rounds"]):
        print(f"\nFederation Round {round_idx + 1}")
        
        # Local training on each client
        client_updates = []
        lamb = conf.get("fedmix_lambda", 0.2)  # FedMix interpolation strength
        
        for client_idx, client in enumerate(clients):
            print(f"\nTraining Client {client_idx + 1}")
            client_update = client.local_train(server.global_model, lamb)
            client_updates.append(client_update)
        
        # Server aggregation
        global_model = server.aggregate(client_updates)
        
        # Optional testing
        if test_loader is not None:
            test_acc = evaluate_model(global_model, test_loader)
            print(f"\nRound {round_idx + 1} Test Accuracy: {test_acc:.2f}%")
    
    return global_model

@torch.no_grad()
def evaluate_model(model, test_loader):
    """Helper function to evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
            
        _, output = model(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total += target.size(0)
    
    return 100.0 * (float(correct) / float(total))