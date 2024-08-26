import os
import cv2
import lpips
import numpy as np
from tqdm import tqdm
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as ssim

# Compute SSIM score
def calculate_ssim(reference_dir, generated_dir):
    ssim_score_list = []

    files = os.listdir(reference_dir)

    for file in tqdm(files):
        if os.path.exists(os.path.join(generated_dir, file)):
            # Load images
            img_true = cv2.imread(os.path.join(reference_dir, file))
            img_fake = cv2.imread(os.path.join(generated_dir, file))
            
            ssim_value = ssim(
                        img_true, img_fake, 
                        channel_axis=-1,
                        gaussian_weights=True, 
                        use_sample_covariance=False, 
                        data_range=img_fake.max() - img_fake.min(),
                        full=False)
        
            ssim_score_list.append(ssim_value)

    return np.mean(ssim_score_list)

# Calculate LPIPS score
def calculate_lpips(dir0, dir1, version='0.1', use_gpu=True):
    # ('-d0','--dir0', type=str, default='./imgs/ex_dir0')
    # ('-d1','--dir1', type=str, default='./imgs/ex_dir1')
    # ('-o','--out', type=str, default='./imgs/example_dists.txt')
    # ('-v','--version', type=str, default='0.1')
    # ('--use_gpu', action='store_true', help='turn on flag to use GPU')
    lpips_score_list = []

    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex', version=version)
    if use_gpu:
        loss_fn.cuda()

    # crawl directories
    files = os.listdir(dir0)

    for file in tqdm(files):
        if os.path.exists(os.path.join(dir1, file)):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(dir0, file)))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(dir1, file)))

            if use_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0, img1).item()
            lpips_score_list.append(dist01)

    return np.mean(lpips_score_list)

# Calculate FID score using PyTorch-FID
def calculate_fid(real_images_folder, fake_images_folder, device = 'cuda'):
    pytorch_fid = fid_score.calculate_fid_given_paths([real_images_folder, fake_images_folder], batch_size=50, device=device, dims=2048)
    return pytorch_fid

if __name__ == '__main__':
    real_images_folder = './results/for_evaluate/img/real'
    fake_images_folder = './results/for_evaluate/img/fake'
    
    # Calculate SSIM score
    print("Calculating SSIM score")
    ssim_score = calculate_ssim(real_images_folder, fake_images_folder)
    
    # Calculate LPIPS score
    print("Calculating LPIPS score")
    lpips_score = calculate_lpips(real_images_folder, fake_images_folder, use_gpu=False)

    # Calculate FID scores
    print("Calculating FID scores")
    pytorch_fid_score = calculate_fid(real_images_folder, fake_images_folder)
    
    # Show scores
    print(f'='*40)
    print(f'SSIM score: {ssim_score}')
    print(f'LPIPS score: {lpips_score}')
    print(f'Pytorch FID score: {pytorch_fid_score}')
    print(f'='*40)