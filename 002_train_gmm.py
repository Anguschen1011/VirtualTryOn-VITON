import tqdm
import torch
import config
import datasets
import tools.utils as utils
import torch.nn.functional as F

from tools.logging import Logger
from tools.loss import LossOperator
from tools.lr_scheduler import WarmupMultiStepLR
from models.model_gmm import CAFWM, LightweightGenerator


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.args.stage = 'GMM'

        self.args = config.MetricsInit(self.args)
        self.logger = Logger(self.args, self.args.SaveFolder)
        self.lossOp = LossOperator(self.args)
        self.visualizer = utils.Visualizer(self.args, self.args.SaveFolder)
        self._build_model()
        self._get_dataloader()
        self._set_optimizer()
    
    def _set_optimizer(self):
        
        # Setting optimizer for Wnet
        self.optimizer_wnet = torch.optim.AdamW(
            [{"params": self.wnet.parameters()}], 
            lr=self.args.lr * 0.5, betas=[self.args.beta1, 0.999]
        )
        self.scheduler_wnet = WarmupMultiStepLR(
            self.optimizer_wnet,
            milestones = [120, 145],
            warmup_iters  = 20,  
            warmup_factor = 1.0 / 3,
            warmup_method = "linear"
        )
        
        # Setting optimizer for Gnet
        self.optimizer_gnet = torch.optim.AdamW(
            [{"params": self.gnet.parameters()}], 
            lr=self.args.lr, betas=[self.args.beta1, 0.999]
        )
        self.scheduler_gnet = WarmupMultiStepLR(
            self.optimizer_gnet,
            milestones = [115, 140],
            warmup_iters  = 15,
            warmup_factor = 1.0 / 3,
            warmup_method = "linear"
        )
    
    def _get_dataloader(self):
        train_dataset = datasets.ImagesDataset(self.args, phase='train')
        self.train_dataLoader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.BatchSize, shuffle=True, persistent_workers=True, pin_memory=True, num_workers=self.args.NumWorkers)
        self.args.len_dataset = len(self.train_dataLoader)

    def _build_model(self):
        self.wnet = CAFWM(self.args).to(self.args.device)
        self.gnet = LightweightGenerator().to(self.args.device)
        self.logger.print_network(self.wnet, 'wnet')
        self.logger.print_network(self.gnet, 'gnet')
        utils.weights_initialization(self.wnet, 'xavier', gain=1.0)
        utils.weights_initialization(self.gnet, 'xavier', gain=1.0)

    def _visualize(self, vis_list, order):
        de_norm = lambda x: (x + 1) / 2 * 255.
        vis_list = [de_norm(image).detach().cpu() for image in vis_list]
        self.visualizer.vis_train(vis_list, order)
                           
    def _train_one_epoch(self):
        self.wnet.train()
        self.gnet.train()
        num_layers = 4
        with tqdm.tqdm(self.train_dataLoader, desc="Training") as pbar:
            for idx, sample in enumerate(pbar):
                image               = sample['image'].to(self.args.device, non_blocking=True)
                cloth               = sample['cloth'].to(self.args.device, non_blocking=True)
                cloth_mask          = sample['cloth_mask'].to(self.args.device, non_blocking=True)
                cloth_agnostic      = sample['agnostic'].to(self.args.device, non_blocking=True)
                person_shape        = sample['person_shape'].to(self.args.device, non_blocking=True)
                person_clothes      = sample['person_clothes'].to(self.args.device, non_blocking=True)
                person_clothes_mask = sample['person_clothes_mask'].to(self.args.device, non_blocking=True)
                
                gloss_mask      = 0
                gloss_tryon     = 0
                gloss_styles    = 0
                gloss_contents  = 0
                
                wloss_styles    = 0
                wloss_contents  = 0
                wloss_cloth     = 0
                wloss_smooth    = 0
                wloss_laplacian = 0
                
                with torch.set_grad_enabled(True):
                    
                    output = self.wnet(cloth, cloth_mask, person_shape)
                    
                    for i in range(num_layers):
                        warping_mask     = output['warping_masks'][i]
                        warped_cloth     = output['warping_cloths'][i]
                        cloth_last_flow  = output['cloth_last_flows'][i]
                        shape_delta_flow = output['shape_delta_flows'][i]


                        _, _, h, w = warping_mask.shape
                        person_clothes_      = F.interpolate(person_clothes     , size=(h, w), mode='nearest')
                        person_clothes_mask_ = F.interpolate(person_clothes_mask, size=(h, w), mode='nearest')

                        # gmm
                        loss_content, loss_style = self.lossOp.calc_vgg_loss(warped_cloth, person_clothes_)
                        wloss_styles    += loss_style * (i+1) * 10
                        wloss_contents  += loss_content * (i+1) * 0.2

                        wloss_cloth     += self.lossOp.criterion_L1(warped_cloth, person_clothes_) * (i+1)
                        wloss_cloth     += self.lossOp.criterion_L1(warping_mask, person_clothes_mask_) * (i+1)
                        wloss_laplacian += self.lossOp.calc_laplacian_loss(cloth_last_flow) * (i+1) * 6
                        wloss_smooth    += self.lossOp.calc_total_variation_loss(shape_delta_flow) * 1.0
                    
                    # try-on
                    warped_cloth = output['warping_cloths'][-1]
                    warping_mask = output['warping_masks'][-1]

                    gen_outputs = self.gnet(warped_cloth, cloth_agnostic)
                    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
                    p_rendered  = F.tanh(p_rendered) 
                    m_composite = F.sigmoid(m_composite) 
                    m_composite = m_composite * warping_mask
                    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
                    
                    loss_content, loss_style = self.lossOp.calc_vgg_loss(p_tryon, image)
                    gloss_styles   += loss_style * (i+1) * 10
                    gloss_contents += loss_content * (i+1) * 0.2

                    loss_content, loss_style = self.lossOp.calc_vgg_loss(p_rendered, image)
                    gloss_styles   += loss_style * (i+1) * 10
                    gloss_contents += loss_content * (i+1) * 0.2

                    gloss_tryon += self.lossOp.criterion_L1(p_tryon   , image) * (i+1)
                    gloss_tryon += self.lossOp.criterion_L1(p_rendered, image) * (i+1)
                    gloss_mask  += torch.mean(torch.abs(1 - m_composite))

                    wloss_cloth     = wloss_cloth * 0.5
                    wloss_smooth    = wloss_smooth * 0.5
                    wloss_laplacian = wloss_laplacian * 0.5
                    gloss_tryon     = gloss_tryon * 0.5
                    gloss_styles    = gloss_styles * 0.5
                    gloss_contents  = gloss_contents * 0.5

                    loss_gmm = wloss_laplacian + wloss_smooth + wloss_cloth + wloss_contents + wloss_styles
                    loss_gen = gloss_tryon + gloss_contents + gloss_styles + gloss_mask
                    
                    loss = loss_gmm * 0.8 + loss_gen

                    self.optimizer_wnet.zero_grad()
                    self.optimizer_gnet.zero_grad()
                    
                    loss.backward()
                    
                    self.optimizer_wnet.step()
                    self.optimizer_gnet.step()
                    wnet_lr = self.scheduler_wnet.get_last_lr()[0]
                    gnet_lr = self.scheduler_gnet.get_last_lr()[0]

                if idx % 1000 == 0:
                    self._visualize([cloth, warped_cloth, person_clothes], order=1)    
                    self._visualize([warped_cloth, (warped_cloth + image)*0.5, p_tryon], order=2)
                    
                self.logger.loss_tmp['wloss-content'] += wloss_contents.item()
                self.logger.loss_tmp['wloss-style']   += wloss_styles.item()
                self.logger.loss_tmp['wloss-lapla']   += wloss_laplacian.item()
                self.logger.loss_tmp['wloss-smooth']  += wloss_smooth.item()
                self.logger.loss_tmp['wloss-cloth']   += wloss_cloth.item()
                self.logger.loss_tmp['gloss-mask']    += gloss_mask.item()
                self.logger.loss_tmp['gloss-tryon']   += gloss_tryon.item()
                self.logger.loss_tmp['gloss-content'] += gloss_contents.item()
                self.logger.loss_tmp['gloss-style']   += gloss_styles.item()
                self.logger.loss_tmp['loss-total']    += loss.item()
                
                pbar.set_description(f"Epoch: {self.args.epoch}, W-net LR: {wnet_lr:.2e}, G-net LR: {gnet_lr:.2e}, Loss: {self.logger.loss_tmp['loss-total'] / (idx + 1):.4f}")
                
            self.logger.loss_tmp = {k : v / self.args.len_dataset for k, v in self.logger.loss_tmp.items()}
            
        return self.logger.loss_tmp['loss-total']


    def train(self):
        
        print(self.args)
        
        for epoch in range(0, self.args.epochs):
            utils.flush()
            
            self.args.epoch = epoch + 1

            loss = self._train_one_epoch()

            self.scheduler_wnet.step()
            self.scheduler_gnet.step()
            
            self.logger.Log_PerEpochLoss()
            self.logger.Reset_PerEpochLoss()
            
            self.visualizer.plot_loss(self.logger.loss_history)

            utils.save_checkpoint(self.args.RootCheckpoint, {
                'args': self.args, 'GMM_state_dict': self.wnet.state_dict(), 'GEN_state_dict': self.gnet.state_dict(),
            }, loss < self.args.best_loss)

            self.args.best_loss = min(loss, self.args.best_loss)


if __name__ == '__main__':
    trainer = Trainer(config.GetConfig())
    trainer.train()
