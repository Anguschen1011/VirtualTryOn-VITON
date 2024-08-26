import torch
import os, cv2, tqdm
import config, datasets
import torch.nn.functional as F

from models.model_gmm import CAFWM
from models.model_gmm import LightweightGenerator

class ModelLearning():
    def __init__(self, args):
        self.args = args
        self.args = config.MetricsInit(self.args)
        self.GetDataloader()

    def GetDataloader(self):
        test_dataset = datasets.ImagesDataset(self.args, phase='test')
        self.test_dataLoader  = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, persistent_workers=True, pin_memory=True, num_workers=self.args.NumWorkers)
        self.args.len_dataset = len(self.test_dataLoader)

    def de_norm(self, x):
        return (x + 1) / 2 * 255.

    def generated_images(self):
        self.net_warp.eval()
        self.net_gen.eval()
        count = 0
        with tqdm.tqdm(self.test_dataLoader, desc="Testing") as pbar:
            for _, sample in enumerate(pbar):
                image        = sample['image'].to(self.args.device)
                cloth        = sample['cloth'].to(self.args.device)
                cloth_mask   = sample['cloth_mask'].to(self.args.device)
                agnostic     = sample['agnostic'].to(self.args.device)
                person_shape = sample['person_shape'].to(self.args.device)

                with torch.set_grad_enabled(False):
                    # Warping Network
                    output = self.net_warp(cloth, cloth_mask, person_shape)
                    # Generative Network
                    warped_mask = output['warping_masks'][-1]
                    warped_cloth = output['warping_cloths'][-1]
                    gen_outputs = self.net_gen(warped_cloth, agnostic)
                    # Create final TryOn image
                    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
                    p_rendered = F.tanh(p_rendered)
                    m_composite = F.sigmoid(m_composite)
                    m_composite = m_composite * warped_mask
                    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

                for i in range(len(image)):
                    count = count + 1

                    cv2.imwrite(f'./results/for_evaluate/img/real/{count:06d}.png'    , self.de_norm(image[i].permute(1, 2, 0).cpu().detach().numpy()))
                    cv2.imwrite(f'./results/for_evaluate/img/fake/{count:06d}.png'    , self.de_norm(p_tryon[i].permute(1, 2, 0).cpu().detach().numpy()))
                    
                    cv2.imwrite(f'./results/for_evaluate/cloth/{count:06d}.png'       , self.de_norm(cloth[i].permute(1, 2, 0).cpu().detach().numpy()))
                    cv2.imwrite(f'./results/for_evaluate/cloth_mask/{count:06d}.png'  , self.de_norm(cloth_mask[i].permute(1, 2, 0).cpu().detach().numpy()))
                    cv2.imwrite(f'./results/for_evaluate/warped_cloth/{count:06d}.png', self.de_norm(warped_cloth[i].permute(1, 2, 0).cpu().detach().numpy()))
                    cv2.imwrite(f'./results/for_evaluate/warped_mask/{count:06d}.png' , self.de_norm(warped_mask[i].permute(1, 2, 0).cpu().detach().numpy()))
                    cv2.imwrite(f'./results/for_evaluate/agnostic/{count:06d}.png'    , self.de_norm(agnostic[i].permute(1, 2, 0).cpu().detach().numpy()))

    def test(self):
        
        if torch.cuda.is_available():
            weight = torch.load(os.path.join(self.args.RootCheckpoint, 'checkpoint.best.pth.tar'))

        weight = torch.load(os.path.join(self.args.RootCheckpoint, 'checkpoint.best.pth.tar'), map_location=torch.device('cpu'))
        self.net_warp = CAFWM(self.args).to(self.args.device)
        self.net_gen  = LightweightGenerator().to(self.args.device)
        self.net_gen.load_state_dict(weight['GEN_state_dict'])
        self.net_warp.load_state_dict(weight['GMM_state_dict'])

        os.makedirs('./results/for_evaluate/img/real', exist_ok=True)
        os.makedirs('./results/for_evaluate/img/fake', exist_ok=True)

        os.makedirs('./results/for_evaluate/cloth'       , exist_ok=True)
        os.makedirs('./results/for_evaluate/cloth_mask'  , exist_ok=True)
        os.makedirs('./results/for_evaluate/warped_cloth', exist_ok=True)
        os.makedirs('./results/for_evaluate/warped_mask' , exist_ok=True)
        os.makedirs('./results/for_evaluate/agnostic'    , exist_ok=True)

        self.generated_images()

if __name__ == '__main__':
    modelLearning = ModelLearning(config.GetConfig())
    modelLearning.test()