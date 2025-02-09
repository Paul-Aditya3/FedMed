import torch
import torch.utils.data
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

class OrganMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.images = df['image'].values
        self.labels = df['label'].values
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()  # Convert numpy array to tensor
        label = torch.tensor(self.labels[idx]).long()
        return image, label

class Client(object):
    def __init__(self, conf, model, train_df):
        self.conf = conf
        self.local_model = model
        self.train_df = train_df
        
        # Create dataset using our custom OrganMNISTDataset class
        self.train_dataset = OrganMNISTDataset(self.train_df)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,

            batch_size=conf["batch_size"], 
            shuffle=True,
            drop_last=True
        )

    def calculate_mean_data(self, mean_batch: int):
        data, label = [], []
        for X, y in self.train_dataset:
            data.append(X)
            label.append(y)
        data = torch.stack(data, dim=0)
        label = torch.stack(label, dim=0)

        random_ids = torch.randperm(len(data))
        data, label = data[random_ids], label[random_ids]
        data = torch.split(data, mean_batch)
        label = torch.split(label, mean_batch)

        self.Xmean, self.ymean = [], []
        for d, l in zip(data, label):
            self.Xmean.append(torch.mean(d, dim=0))
            self.ymean.append(torch.mean(F.one_hot(l, num_classes=self.conf["num_classes"][self.conf["which_dataset"]]).to(dtype=torch.float32), dim=0))
        self.Xmean = torch.stack(self.Xmean, dim=0)
        self.ymean = torch.stack(self.ymean, dim=0)
        return self.Xmean, self.ymean

    def get_mean_data(self, Xg, Yg):
        self.Xg = Xg
        self.Yg = Yg

    def local_train(self, model, lamb):
        """
        Local training with proper batch size handling
        """
        
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'], weight_decay=self.conf["weight_decay"])
        criterion = torch.nn.CrossEntropyLoss()
        
        
        if len(self.Yg.shape) == 3 and self.Yg.shape[1] == 1:
            self.Yg = self.Yg.squeeze(1)  
        
        for e in range(self.conf["local_epochs"]):
            self.local_model.train()
            epoch_loss = 0
            batch_count = 0

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                target = target.squeeze() 
                
                current_batch_size = data.size(0)
   
                inputX = (1 - lamb) * data   
                inputX.requires_grad_()

                
                idg = torch.randint(len(self.Xg), (1,)).item()
                
                
                xg = self.Xg[idg]  
                if len(xg.shape) == 3:  
                    xg = xg.unsqueeze(0)  
                xg = xg.expand(current_batch_size, -1, -1, -1)  

                
                yg = self.Yg[idg]  
                if len(yg.shape) > 1: 
                    yg = yg.squeeze()  
                yg = yg.unsqueeze(0)  
                yg = yg.expand(current_batch_size, -1) 

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                    inputX = inputX.cuda()
                    xg = xg.cuda()
                    yg = yg.cuda()

                optimizer.zero_grad()
                features, output = self.local_model(inputX)
                

                
                loss1 = (1 - lamb) * criterion(output, target)
                loss2 = lamb * criterion(output, torch.argmax(yg, dim=1))

                gradients = autograd.grad(outputs=loss1, inputs=inputX,
                                        create_graph=True, retain_graph=True)[0]
                
                
                gradients_flat = gradients.view(current_batch_size, -1)
                xg_flat = xg.view(current_batch_size, -1)
                loss3 = lamb * torch.mean(torch.sum(gradients_flat * xg_flat, dim=1))

                loss = loss1 + loss2 + loss3
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_epoch_loss = epoch_loss / batch_count
            acc, eval_loss = self.model_eval()
            print(f"Epoch {e+1} done. train_loss={avg_epoch_loss:.4f}, train_acc={acc:.2f}")

        return self.local_model.state_dict()


    @torch.no_grad()
    def model_eval(self):
        self.local_model.eval()
        
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        for batch_id, batch in enumerate(self.train_loader):
            data, target = batch
            target = target.squeeze()  # Add squeeze here <<<<<<<<<<
            
            dataset_size += data.size()[0]
            
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                
            _, output = self.local_model(data)
            
            total_loss += criterion(output, target)
            pred = output.data.max(1)[1]
            
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss.cpu().detach().numpy() / dataset_size
        
        return acc, total_l