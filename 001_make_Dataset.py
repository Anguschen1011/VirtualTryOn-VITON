import sys
import pickle
import config
import os
import os.path as osp
from tqdm import tqdm
sys.path.append('.')


class AnnotationBuilding():
    def __init__(self, args):
        self.args = args

    def CreateAnnotationFile(self):
        def pickup_data(phase):
            # load data list
            samples = []
            with open(osp.join(self.args.DatasetRoot, f"{phase}_pairs.txt"), 'r') as f:
                for line in tqdm(f.readlines()):
                    im_name, c_name = line.strip().split()
                    im_name_png     = im_name.replace('.jpg', '.png')
                    pose_name       = im_name.replace('.jpg', '_keypoints.json')

                    pose_file        = osp.join(self.args.DatasetRoot, phase, 'pose', pose_name)
                    image_file       = osp.join(self.args.DatasetRoot, phase, 'image', im_name)
                    cloth_file       = osp.join(self.args.DatasetRoot, phase, 'cloth', c_name)
                    self_cloth_mask  = osp.join(self.args.DatasetRoot, phase, 'cloth-mask-self', c_name)
                    self_parse_image = osp.join(self.args.DatasetRoot, phase, 'parsing_atr', im_name_png)
                    
                    if not os.path.isfile(pose_file):
                        continue
                    if not os.path.isfile(image_file):
                        continue
                    if not os.path.isfile(cloth_file):
                        continue
                    
                    samples.append({
                        'cloth_name'  : c_name, 
                        'image_name'  : im_name,
                        'cloth'       : cloth_file, 
                        'cloth_mask'  : self_cloth_mask,
                        'image'       : image_file,
                        'pose_label'  : pose_file,
                        'parse_image' : self_parse_image,
                    })
            
            return samples
        
        training_sample = pickup_data('train')
        testing_sample  = pickup_data('test')
        data_file       = {'Training_Set':training_sample, 'Testing_Set':testing_sample,}
        
        pickle.dump(data_file, open(self.args.AnnotFile, 'wb'))
        
        print(training_sample[0])
        print(testing_sample[0])

if __name__== '__main__':
    annotationBuilding = AnnotationBuilding(config.GetConfig())
    os.makedirs('./data', exist_ok=True)
    annotationBuilding.CreateAnnotationFile()

