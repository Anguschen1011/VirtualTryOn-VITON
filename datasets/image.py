import cv2
import sys
import json
import torch
import pickle
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
sys.path.append('.')

"""
bgr_color_atr = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192], [128, 128, 64], 
[128, 128, 192], [0, 64, 0], [0, 64, 128]]

atr_rgb = {'Background': [0, 0, 0], 'Hat': [128, 0, 0], 'Hair': [0, 128, 0], 'Sunglasses': [128, 128, 0], 'Upper-clothes': [0, 0, 128], 'Skirt': [128, 0, 128], 'Pants': [0, 128, 128], 'Dress': [128, 128, 128],
'Belt': [64, 0, 0], 'Left-shoe': [192, 0, 0], 'Right-shoe': [64, 128, 0], 'Face': [192, 128, 0], 'Left-leg': [64, 0, 128], 'Right-leg': [192, 0, 128], 'Left-arm': [64, 128, 128], 'Right-arm': [192, 128, 128],
'Bag': [0, 64, 0], 'Scarf': [128, 64, 0]}

bgr_color_lip = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192], [128, 128, 64], 
[128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128]]

lip_rgb = {'Background': [0, 0, 0], 'Hat': [128, 0, 0], 'Hair': [0, 128, 0], 'Glove': [128, 128, 0], 'Sunglasses': [0, 0, 128], 'Upper-clothes': [128, 0, 128], 'Dress': [0, 128, 128], 'Coat': [128, 128, 128],
'Socks': [64, 0, 0], 'Pants': [192, 0, 0], 'Jumpsuits': [64, 128, 0], 'Scarf': [192, 128, 0], 'Skirt': [64, 0, 128], 'Face': [192, 0, 128], 'Left-arm': [64, 128, 128], 'Right-arm': [192, 128, 128], 'Left-leg': [0, 64, 0],
'Right-leg': [128, 64, 0], 'Left-shoe': [0, 192, 0], 'Right-shoe': [128, 192, 0]}
"""

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, args, phase='train'):
        self.args = args
        self.phase = phase
        self.fine_width  = self.args.fine_width
        self.fine_height = self.args.fine_height
        self._read_path_label()
        self._setup_transforms()
        self.image_grid = cv2.imread('./grid.png')
        self.image_grid = self.transform_bin(image=self.image_grid)["image"]
        self.image_grid = self.from_255_to_norm(self.image_grid)

    def _read_path_label(self):
        assert self.phase in ['train', 'test'], 'Phase must be in : Train or Test'
        pkl = pickle.load(open(self.args.AnnotFile, 'rb'))
        if self.phase == 'train':
            self.data = pkl['Training_Set']
        elif self.phase == 'test':
            self.data = pkl['Testing_Set']
        self.dataset_size = len(self.data)

    def _setup_transforms(self):
        self.transform_bin = A.Compose([
            A.Resize(self.fine_height, self.fine_width), 
            ToTensorV2()
        ])

    def load_keypoints(self, pose_file):
        with open(pose_file, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
        return pose_data.astype(np.float32)

    def draw_image(self, pose_data):
        r = self.args.radius
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        image_pose = np.zeros((self.fine_height, self.fine_width))
        for i in range(point_num):
            one_map = np.zeros((self.fine_height, self.fine_width)).astype(np.float32)
            pointx = int(pose_data[i, 0])
            pointy = int(pose_data[i, 1])
            if pointx > 1 and pointy > 1:
                cv2.rectangle(one_map, (pointx-r, pointy-r), (pointx+r, pointy+r), (255, 255, 255), -1)
                cv2.rectangle(image_pose, (pointx-r, pointy-r), (pointx+r, pointy+r), (255, 255, 255), -1)
            one_map = self.transform_bin(image=one_map)["image"]                      # [-1, 1]
            one_map = self.from_255_to_norm(one_map)
            pose_map[i] = one_map[0]
        image_pose = self.transform_bin(image=image_pose)["image"]                    # [-1, 1]
        image_pose = self.from_255_to_norm(image_pose)
        return image_pose, pose_map
    
    def self_parsing_mask(self, image, BGR_color):
        
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        for color in BGR_color:
            lower = np.array(color)
            upper = np.array(color)
            
            color_mask = cv2.inRange(image, lower, upper)
            
            mask[color_mask > 0] = 0
        
        # Use dilation to expand the white area
        kernel = np.ones((2, 2), np.uint8)
        mask   = cv2.dilate(mask, kernel, iterations=1)
        
        # Use GaussianBlur to make the edges smoother
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        mask = torch.tensor(mask, dtype=torch.float32) # [256, 192]
        mask = mask.unsqueeze(0)                       # [1, 256, 192]

        return mask / 255 # [0, 1]
    
    def from_255_to_norm(self, x):
        return x / 255 * 2 - 1
    
    def to_norm(self, x):
        return x * 2 - 1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx, method = 'lip'):
        
        image            = cv2.imread(self.data[idx]['image'])
        cloth            = cv2.imread(self.data[idx]['cloth'])
        cloth_mask_self  = cv2.imread(self.data[idx]['cloth_mask'], 0)
        self_parse_image = cv2.imread(self.data[idx]['parse_image'])
        pose_data        = self.load_keypoints(self.data[idx]['pose_label'])

        method = 'atr'
        
        if method == 'lip':
            ### LIP Method ###
            # Keep Hat, Hair, Sunglasses, Face white, else turn into black.
            face_bgr_color   = [[0, 0, 0], [0, 128, 128], [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], 
                                [0, 128, 192], [128, 0, 64], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128]]

            # Keep Upper-clothes white, else turn into black. Remove Dress?
            person_cloth_bgr_color = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192],[0, 128, 64],
                                        [0, 128, 192], [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128]]

            # Keep Socks, Pants, Jumpsuits, Skirt, Left-leg, Right-leg, Left-shoe, Right-shoe white, else turn into black. Add Dress?
            others_bgr_color = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 128, 0], [128, 0, 128], [128, 128, 128], 
                                [0, 128, 192], [128, 0, 192], [128, 128, 64], [128, 128, 192]]

            agnostic_bgr_color  = [[0, 0, 0], [128, 0, 128], [128, 128, 0]] # Turn Background, Upper-clothes into Black. Add Dress?

        if method == 'atr':
            ### ATR Method ###
            """
            atr_rgb = {'Background': [0, 0, 0], 'Hat': [128, 0, 0], 'Hair': [0, 128, 0], 'Sunglasses': [128, 128, 0], 'Upper-clothes': [0, 0, 128], 'Skirt': [128, 0, 128],
            'Pants': [0, 128, 128], 'Dress': [128, 128, 128],'Belt': [64, 0, 0], 'Left-shoe': [192, 0, 0], 'Right-shoe': [64, 128, 0], 'Face': [192, 128, 0], 'Left-leg': [64, 0, 128],
            'Right-leg': [192, 0, 128], 'Left-arm': [64, 128, 128], 'Right-arm': [192, 128, 128],'Bag': [0, 64, 0], 'Scarf': [128, 64, 0]}
            """
            # Keep Hat, Hair, Sunglasses, Face white, else turn into black.
            face_bgr_color = [[0, 0, 0], [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64],
                                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128]]

            # Keep Upper-clothes white, else turn into black. Remove Dress?
            person_cloth_bgr_color = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 128], [128, 128, 0], [0, 0, 64], [0, 0, 192],
                                    [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128]]

            # Keep Pants, Jumpsuits, Skirt, Left-leg, Right-leg, Left-shoe, Right-shoe white, else turn into black. Add Dress?
            others_bgr_color = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 128, 128], [0, 0, 64], [0, 128, 192], [128, 128, 64],
                                [128, 128, 192], [0, 64, 0], [0, 64, 128]]

            agnostic_bgr_color  = [[0, 0, 0], [128, 0, 0], [128, 128, 128]] # Turn Background, Upper-clothes into Black. Add Dress?
        

        image_pose, pose_map = self.draw_image(pose_data)                       # [-1, 1]   # torch.tensor

        cloth_mask_self = (cloth_mask_self / 255.).astype(np.float32)           # [0, 1]
        cloth = cloth * np.expand_dims(cloth_mask_self, -1)                                         # [0, 255]
        
        parse_face_mask   = self.self_parsing_mask(self_parse_image, face_bgr_color)
        person_cloth_mask = self.self_parsing_mask(self_parse_image, person_cloth_bgr_color)
        parse_others_mask = self.self_parsing_mask(self_parse_image, others_bgr_color)
        
        preserve_mask  = torch.cat([parse_face_mask, parse_others_mask], dim=0) # [0, 1]        

        image          = self.transform_bin(image=image)["image"]               # [0, 255]
        cloth          = self.transform_bin(image=cloth)["image"]               # [0, 255]

        person_clothes = image * person_cloth_mask
        
        agnostic_mask = self.self_parsing_mask(self_parse_image, agnostic_bgr_color)
        agnostic = image * agnostic_mask
        
        cloth_mask_self   = torch.from_numpy(cloth_mask_self).unsqueeze(0)      # [0, 1]

        image             = self.from_255_to_norm(image)          # [-1, 1]
        cloth             = self.from_255_to_norm(cloth)          # [-1, 1]
        person_clothes    = self.from_255_to_norm(person_clothes) # [-1, 1]
        agnostic          = self.from_255_to_norm(agnostic)       # [-1, 1]
        preserve_mask     = self.to_norm(preserve_mask)           # [-1, 1]
        cloth_mask_self   = self.to_norm(cloth_mask_self)         # [-1, 1]
        person_cloth_mask = self.to_norm(person_cloth_mask)       # [-1, 1]
   
        person_shape  = torch.cat([agnostic, preserve_mask, pose_map], dim=0)    # [-1, 1]   

        return {
            'cloth_name'          : self.data[idx]['cloth_name'],   # for visualization
            'image_name'          : self.data[idx]['image_name'],   # for visualization or ground truth
            'image'               : image,                          # (192, 256, 3 ) for visualization
            'cloth'               : cloth,                          # (192, 256, 3 ) for input
            'cloth_mask'          : cloth_mask_self,                # (192, 256, 1 ) for input
            'agnostic'            : agnostic,                       # (192, 256, 22) for input
            'person_shape'        : person_shape,                   # (192, 256, 22) for input
            'person_clothes'      : person_clothes,                 # (192, 256, 3 ) for ground trut
            'person_clothes_mask' : person_cloth_mask,              # (192, 256, 1 ) for ground truth
        }
        
    
class data_prefetcher():
    def __init__(self, args, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(args.device)
        self.preload()

    def preload(self):
        try:
            self.next_image, self.next_label = next(self.loader)
        except StopIteration:
            self.next_image = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image = self.next_image
        label = self.next_label 
        self.preload()
        return image, label