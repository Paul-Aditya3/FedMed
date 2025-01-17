import torch
from setup import setup_fedmix
from server import Server, train_fedmix  

def main():
 
    print("Setting up FedMix components...")
    conf, model, clients, test_loader = setup_fedmix()
    print(f"Setup completed with {len(clients)} clients")
    
    
    print("\nStarting FedMix training...")
    final_model = train_fedmix(conf, model, clients, test_loader)
    
    
    torch.save(final_model.state_dict(), 'fedmix_model.pth')
    print("\nTraining completed and model saved to 'fedmix_model.pth'")

if __name__ == "__main__":
    main()