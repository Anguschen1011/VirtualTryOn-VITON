import torch 

class OptimOperater():
    def __init__(self, args, G_params, D_params):
        self.args = args
        self.GetOptimizer(G_params, D_params)
        self.GetScheduler()
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()


    def GetOptimizer(self, G_params, D_params):
        self.G_optimizer = torch.optim.AdamW(G_params, lr=self.args.lr_G, betas=[self.args.Beta1, self.args.Beta2])
        self.D_optimizer = torch.optim.AdamW(D_params, lr=self.args.lr_D, betas=[self.args.Beta1, self.args.Beta2])
        
    
    def GetScheduler(self):
        self.G_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.G_optimizer, T_0=self.args.T_0, eta_min=self.args.lr_G * self.args.lr_Decay_Factor
        )
        self.D_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.D_optimizer, T_0=self.args.T_0, eta_min=self.args.lr_D * self.args.lr_Decay_Factor
        )


    def OptimizeModel(self, loss, flag):
        optimizer = self.__getattribute__(f'{flag}_optimizer')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def AdjustLR(self, flag):
        scheduler = self.__getattribute__(f'{flag}_scheduler')
        scheduler.step()


