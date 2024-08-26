import os, logging, time


class Logger():
    def __init__(self, args, SaveFolder_root):
        self.args = args
        self.SaveFolder = SaveFolder_root
        self.init_loss()
        self.CreateLogger()
        self.Reset_PerEpochLoss()

    def init_loss(self):
        self.loss_list = [
            'wloss-cloth', 'wloss-content', 'wloss-style', 'wloss-lapla', 'wloss-smooth', 
            'gloss-tryon', 'gloss-content', 'gloss-style', 'gloss-mask',
            'loss-total',
        ]  
        self.loss_history = {k:[] for k in self.loss_list}

    def CreateLogger(self, console=True):
        os.makedirs(self.SaveFolder, exist_ok=True)
        _level = logging.INFO

        self.logger = logging.getLogger()
        self.logger.setLevel(_level)

        if console:
            cs = logging.StreamHandler()
            cs.setLevel(_level)
            self.logger.addHandler(cs)

        file_name = os.path.join(self.SaveFolder, 'model_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)
        self.logger.addHandler(fh)

    def Reset_PerEpochLoss(self):
        self.start_time = time.time()
        self.loss_tmp = {k:0.0 for k in self.loss_list}

    def Log_PerEpochLoss(self):
        log_str = '\n' + '='*40 + f'\nEpoch {self.args.epoch}, time {time.time() - self.start_time:.2f} s'
        self.logger.info(log_str) if self.logger is not None else print(log_str)
        for k, v in self.loss_tmp.items():
            self.loss_history[k].append(v)
            log_str = f'{k:s}\t{v:.6f}'
            self.logger.info(log_str) if self.logger is not None else print(log_str)  
        self.logger.info('='*40) if self.logger is not None else print('='*40)
    
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        log_str = f'{name:s}, the number of parameters: {num_params:d}'
        self.logger.info(log_str) if self.logger is not None else print(log_str)
