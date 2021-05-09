import os, torch, pdb
import numpy as np
import json
from PIL import Image
from PIL import ImageFile
import torch.utils.data as data
import random#, cv2
import collections
from numpy import random as nprandom
import pickle
import glob
import re
import numpy as np
import pandas as pd
from random import shuffle
import random
import math
#import nibabel.freesurfer.mghformat as mgh
import nibabel as nib
from .augmentations import *
#import utils.augmentations as AUGM
#import utils.normalizations as NORM
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ADNI_3D(data.Dataset):


    def __init__(self, dir_to_scans, dir_to_tsv, mode = 'Train', use_mask = None, mask_type = 'graymatter', n_label = 3, percentage_usage = 1.0):
        if n_label == 3:
            LABEL_MAPPING = [0.0, 0.5, 1.0, 2.0]
        elif n_label == 2:
            LABEL_MAPPING = [0.0, 0.5, 1.0]
        self.LABEL_MAPPING = LABEL_MAPPING     
        if mode == 'Train':
            subject_tsv = pd.io.parsers.read_csv(os.path.join(dir_to_tsv,
                #'Scan_info_NACC_SixMonth.tsv'), sep='\t') 
                mode+'_diagnosis_ADNI_AIBL_NACC_65_85.tsv'), sep='\t')
            '''
            subject_tsv_OASIS = pd.io.parsers.read_csv(os.path.join(dir_to_tsv,
                'All_diagnosis_OASIS.tsv'), sep='\t') 
            subject_tsv_AIBL = pd.io.parsers.read_csv(os.path.join(dir_to_tsv,
                'All_diagnosis_AIBL.tsv'), sep='\t') 
            subject_tsv = subject_tsv[['participant_id', 'session_id', 'diagnosis', 'mmse', 'cdr']].append(subject_tsv_OASIS[['participant_id',
             'session_id', 'diagnosis', 'mmse', 'cdr']]).append(subject_tsv_AIBL[['participant_id', 'session_id', 'diagnosis', 'mmse', 'cdr']])
             '''
                
        else:
            subject_tsv = pd.io.parsers.read_csv(os.path.join(dir_to_tsv,#'All_diagnosis_AIBL.tsv'), sep='\t')
                mode+'_diagnosis_ADNI.tsv'), sep='\t') 
                #'All_diagnosis_NACC_SixMonth_volume.tsv'),sep='\t')

                #mode+'_diagnosis_ADNI_umass.tsv'), sep='\t')

        # Clean sessions without labels
        indices_not_missing = []
        for i in range(len(subject_tsv)):
            if mode == 'Train':
                if (subject_tsv.iloc[i].cdr in LABEL_MAPPING): #and not (np.isnan(subject_tsv.iloc[i].cdr)):
                    indices_not_missing.append(i)
            else:
                if (subject_tsv.iloc[i].cdr in LABEL_MAPPING):
                    indices_not_missing.append(i)

        self.subject_tsv = subject_tsv.iloc[indices_not_missing]
        if mode == 'Train':
            self.subject_tsv = subject_tsv.iloc[np.random.permutation(int(len(subject_tsv)*percentage_usage))]
        self.subject_id = np.unique(subject_tsv.participant_id.values)
        self.index_dic = dict(zip(self.subject_id,range(len(self.subject_id))))
        self.dir_to_scans = dir_to_scans

        self.use_mask = use_mask
        self.mask_type = mask_type
        self.mode = mode
        self.age_range = list(np.arange(0.0,120.0,0.5))



    def __len__(self):
        return len(self.subject_tsv)

    def __getitem__(self, idx):
        try:
            if 'OAS' in self.subject_tsv.iloc[idx].participant_id:
                path = os.path.join('/gpfs/data/razavianlab/skynet/alzheimers/OASIS/oasis3/OASIS3_processed/subjects/',self.subject_tsv.iloc[idx].participant_id,
                    self.subject_tsv.iloc[idx].session_id,'t1/spm/segmentation/normalized_space')
            elif 'AIBL' in self.subject_tsv.iloc[idx].participant_id:
                path = os.path.join('/gpfs/data/razavianlab/data/mri/aibl_all/AIBL_processed/subjects/',self.subject_tsv.iloc[idx].participant_id,
                    self.subject_tsv.iloc[idx].session_id,'t1/spm/segmentation/normalized_space')
            elif 'NACC' in self.subject_tsv.iloc[idx].participant_id:
                path = os.path.join('/gpfs/data/razavianlab/data/mri/nacc_pp_all/NACC_processed_new/subjects/',self.subject_tsv.iloc[idx].participant_id,
                    self.subject_tsv.iloc[idx].session_id,'t1/spm/segmentation/normalized_space')
            else:
                path = os.path.join(self.dir_to_scans,self.subject_tsv.iloc[idx].participant_id,
                    self.subject_tsv.iloc[idx].session_id,'t1/spm/segmentation/normalized_space')
                path_mask = os.path.join('/gpfs/data/razavianlab/skynet/alzheimers/adni_pp_all/ADNI_seg/subjects/',self.subject_tsv.iloc[idx].participant_id,
                    self.subject_tsv.iloc[idx].session_id)
            all_segs = list(os.listdir(path))
            if self.subject_tsv.iloc[idx].cdr == 0:
                label = 0
            elif self.subject_tsv.iloc[idx].cdr == 0.5:
                label = 1
            elif self.subject_tsv.iloc[idx].cdr > 0.5:
                label = 2
                # if self.LABEL_MAPPING == ["CN", "AD"]:
                #     label = 1
                # else:
                #     label = 2
            else:
                print('WRONG LABEL VALUE!!!')
                label = -100
            try:
                mmse = self.subject_tsv.iloc[idx].mmse
            except:
                mmse = 0
            cdr_sub = 0#self.subject_tsv.iloc[idx].cdr #cdr_sb #cdr#
            age = list(np.arange(0.0,120.0,0.5)).index(self.subject_tsv.iloc[idx].age_rounded) #list(np.arange(0.0,25.0)).index(self.subject_tsv.iloc[idx].education_level)#
            age_ori = 0#self.subject_tsv.iloc[idx].age
            #age = self.subject_tsv.iloc[idx].age
            '''
            cdr = self.subject_tsv.iloc[idx].cdr
            if cdr == 0.0:
                cdr_out = 0
            elif cdr == 0.5:
                cdr_out = 1
            else:
                cdr_out = 2
            '''
            idx_out = self.index_dic[self.subject_tsv.iloc[idx].participant_id]

            

            for seg_name in all_segs:
                if 'Space_T1w' in seg_name:
                    image = nib.load(os.path.join(path,seg_name)).get_data().squeeze()
                    image[np.isnan(image)] = 0.0
                    image = (image - image.min())/(image.max() - image.min() + 1e-3)
                    #image = (image - image.mean())/(image.std() + 1e-6)
                    if self.mode == 'Train':
                        image = self.augment_image(image)
                    #image = translateit(image, [10,5,10], isseg=True)
                    #intense_factor = np.random.uniform(0.9,1.1,1)[0]
                    #image = intensifyit(image,intense_factor)
                    #theta = np.random.uniform(-15,15,1)[0]
                    #image = rotateit(image, (2,0), theta, isseg=False)
                    
                    #scale_factor = np.random.uniform(0.8,1.3,1)[0]
                    #image = scaleit(image,scale_factor)
                    #image = flipit(image)
                    #theta = np.random.uniform(-140,140,1)[0]
                    #image = rotateit(image, (1,0), theta, isseg=False)
           
            #for seg_name in all_segs:
                #if self.mask_type in seg_name:
                #    brain_mask = nib.load(os.path.join(path,seg_name)).get_data().squeeze()
            if self.use_mask:
                #image *= brain_mask
                path_mask += '/'+str(self.subject_tsv.iloc[idx].participant_id) +'_'+ str(self.subject_tsv.iloc[idx].session_id)+'_seg.p'
                brain_mask = self.unpickling(path_mask)

                image = np.stack([image,brain_mask], axis=0)
            else:
                image = np.expand_dims(image,axis =0)
            mask_id = 58#random.randint(56,61)
            #image = image[5:-6,10:-10,0:-11]
            if self.mode == 'Train':
                image = self.randomCrop(image,96,96,96)
            else:
                image = self.centerCrop(image,96,96,96)


            

        except Exception as e:
                print(f"Failed to load #{idx}: {path}")
                print(f"Errors encountered: {e}")
                print(traceback.format_exc())
                return None,None,None,None
        return image.astype(np.float32),label, idx, idx, mmse,cdr_sub,age

    def centerCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height

        x = img.shape[1]//2 - length//2
        y = img.shape[2]//2 - width//2
        z = img.shape[3]//2 - height//2
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img

    def randomCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height
        x = random.randint(0, img.shape[1] - length)
        y = random.randint(0, img.shape[2] - width)
        z = random.randint(0, img.shape[3] - height )
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img
    def augment_image(self, image):
        sigma = np.random.uniform(0.0,1.0,1)[0]
        image = scipy.ndimage.filters.gaussian_filter(image, sigma, truncate=8)

        #image = (image - image.min())/(image.max() - image.min() + 1e-6)
        #image = translateit(image, [10,5,10], isseg=True)
        #intense_factor = np.random.uniform(0.9,1.1,1)[0]
        #image = intensifyit(image,intense_factor)
        #theta = np.random.uniform(-15,15,1)[0]
        #image = rotateit(image, (2,0), theta, isseg=False)
        

        #scale_factor = np.random.uniform(0.8,1.3,1)[0]
        #image = scaleit(image,scale_factor)
        #image = flipit(image)
        #theta = np.random.uniform(-140,140,1)[0]
        #image = rotateit(image, (1,0), theta, isseg=False)
        #image = translateit(image, offset）

        return image

    def unpickling(self, path):
       file_return=pickle.load(open(path,'rb'))
       return file_return


# import os, torch, pdb
# import numpy as np
# import json
# from PIL import Image
# from PIL import ImageFile
# import torch.utils.data as data
# import random 
# import collections
# from numpy import random as nprandom
# import pickle
# import glob
# import re
# import numpy as np
# import pandas as pd
# from random import shuffle
# import random
# import math
# import nibabel as nib
# import cv2
# from .augmentations import *
# import mclahe as mc
# import torchio

# from torchio.transforms import (
#     Compose, 
#     RandomFlip,
#     RandomAffine,
#     RandomMotion,
#     RandomGhosting,
#     RandomBiasField,
#     RandomBlur,
#     RandomNoise,
#     RandomSwap,
#     ZNormalization,
#     )


# ImageFile.LOAD_TRUNCATED_IMAGES = True

# class ADNI_3D(data.Dataset):


#     def __init__(self, dir_to_scans, dir_to_tsv, mode = 'Train', n_label = 3, percentage_usage = 1.0):
#         if n_label == 3:
#             LABEL_MAPPING = ["CN", "MCI", "AD"]
#         elif n_label == 2:
#             LABEL_MAPPING = ["CN", "AD"]
#         self.LABEL_MAPPING = LABEL_MAPPING     
#         if mode == 'Train':
#             subject_tsv = pd.io.parsers.read_csv(os.path.join(dir_to_tsv, 
#                 mode+'_diagnosis_ADNI.tsv'), sep='\t')
#         else:
#             subject_tsv = pd.io.parsers.read_csv(os.path.join(dir_to_tsv,
#                 mode+'_diagnosis_ADNI.tsv'), sep='\t') 

#         self.dir_to_scans = dir_to_scans





#         # Clean sessions without labels
#         indices_not_missing = []
#         for i in range(len(subject_tsv)):
#             path_temp = os.path.join(self.dir_to_scans,subject_tsv.iloc[i].participant_id,
#                 subject_tsv.iloc[i].session_id,'t1_linear')
#             ind = os.path.exists(path_temp)
#             if mode == 'Train':
#                 if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING) and ind:
#                     indices_not_missing.append(i)
#             else:
#                 if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING) and ind:
#                     indices_not_missing.append(i)
    

#         self.subject_tsv = subject_tsv.iloc[indices_not_missing]
#         if mode == 'Train':
#             self.subject_tsv = self.subject_tsv.iloc[np.random.permutation(int(len(self.subject_tsv)*percentage_usage))]
#         self.subject_id = np.unique(self.subject_tsv.participant_id.values)
#         self.index_dic = dict(zip(self.subject_id,range(len(self.subject_id))))
        


#         self.mode = mode
#         self.age_range = list(np.arange(0.0,120.0,0.5))

#         if mode == 'Train':

#             self.transform = Compose([
#                                       RandomAffine(scales=(0.9, 1.2),
#                                                    degrees=(10),
#                                                    isotropic=False,
#                                                    default_pad_value='otsu',
#                                                    image_interpolation='bspline',
#                                                    ),
#                                       RandomMotion(p=0.3),
#                                       RandomGhosting(p=0.3),
#                                       RandomBiasField(p=0.3),
#                                       RandomBlur(),
#                                       RandomNoise(),
#                                       #RandomSwap(),
#                                       ZNormalization(),

#                 ])
#         else:
#             self.transform = ZNormalization()



#     def __len__(self):
#         return len(self.subject_tsv)

#     def __getitem__(self, idx):
#         try:
#             path = os.path.join(self.dir_to_scans,self.subject_tsv.iloc[idx].participant_id,
#                 self.subject_tsv.iloc[idx].session_id,'t1_linear')#'t1_bias_correction')#'t1/spm/segmentation/normalized_space')

#             all_segs = list(os.listdir(path))
#             if self.subject_tsv.iloc[idx].diagnosis == 'CN':
#                 label = 0
#             elif self.subject_tsv.iloc[idx].diagnosis == 'MCI':
#                 label = 1
#             elif self.subject_tsv.iloc[idx].diagnosis == 'AD':
#                 if self.LABEL_MAPPING == ["CN", "AD"]:
#                     label = 1
#                 else:
#                     label = 2
#             else:
#                 print('WRONG LABEL VALUE!!!')
#                 label = -100
#             mmse = 0#self.subject_tsv.iloc[idx].mmse
#             cdr_sub = 0#self.subject_tsv.iloc[idx].cdr #cdr_sb #cdr#
#             age = list(np.arange(0.0,120.0,0.5)).index(self.subject_tsv.iloc[idx].age_rounded) #list(np.arange(0.0,25.0)).index(self.subject_tsv.iloc[idx].education_level)#

#             idx_out = self.index_dic[self.subject_tsv.iloc[idx].participant_id]

            

#             for seg_name in all_segs:
#                 if 'Crop_res-1x1x1_T1w' in seg_name:
#                 #if 'BiasCorr_T1w' in seg_name:
#                 #if 'Space_T1w' in seg_name:
#                     image = nib.load(os.path.join(path,seg_name)).get_data().squeeze()
#                     image[np.isnan(image)] = 0.0

#             image = np.expand_dims(image,axis =0)
#             if self.mode == 'Train':
#                 image = self.randomCrop(image,128,128,128)
#                 image = self.transform(image)
#             else:
#                 image = self.centerCrop(image,128,128,128)
#                 image = self.transform(image)
#                 image = np.expand_dims(image,axis =0)
            
#         except Exception as e:
#             print(f"Failed to load #{idx}: {path}")
#             print(f"Errors encountered: {e}")
#             print(traceback.format_exc())
#             return None,None,None,None
#         return image.astype(np.float32),label,idx, idx_out,mmse,cdr_sub,age

#     def Crop(self, img):
#         img = img[:,:, 10:-50, 70:-20]
#         return img

#     def centerCrop(self, img, length, width, height):
#         assert img.shape[1] >= length
#         assert img.shape[2] >= width
#         assert img.shape[3] >= height

#         x = img.shape[1]//2 - length//2
#         y = img.shape[2]//2 - width//2
#         z = img.shape[3]//2 - height//2
#         img = img[:,x:x+length, y:y+width, z:z+height]
#         return img

#     def gridCrop(self, img, length, width, height):
#         assert img.shape[1] >= length
#         assert img.shape[2] >= width
#         assert img.shape[3] >= height

#         indent = 10
#         img1 = img[:,indent:length+indent, indent:width+indent, indent:height+indent]
#         img2 = img[:,img.shape[1] - length - indent:img.shape[1]- indent, indent:width+indent, indent:height+indent]
#         img3 = img[:,indent:length+indent, img.shape[2] - width - indent :img.shape[2]- indent, indent:height+indent]
#         img4 = img[:,indent:length+indent, indent:width+indent, img.shape[3] - height - indent:img.shape[3] - indent]
        
#         img5 = img[:,img.shape[1] - length - indent :img.shape[1] - indent, img.shape[2] - width - indent:img.shape[2] - indent, indent:height+indent]
#         img6 = img[:,img.shape[1] - length - indent :img.shape[1] - indent, indent:width+indent, img.shape[3] - height - indent:img.shape[3] - indent]
#         img7 = img[:,indent:length+indent, img.shape[2] - width - indent:img.shape[2] - indent, img.shape[3] - height - indent :img.shape[3] - indent]
#         img8 = img[:,img.shape[1] - length - indent:img.shape[1] - indent, img.shape[2] - width - indent:img.shape[2] - indent, img.shape[3] - height - indent:img.shape[3] - indent] 


#         img1 = (img1 - img1.min())/(img1.max() - img1.min() + 1e-6)
#         img2 = (img2 - img2.min())/(img2.max() - img2.min() + 1e-6)
#         img3 = (img3 - img3.min())/(img3.max() - img3.min() + 1e-6)
#         img4 = (img4 - img4.min())/(img4.max() - img4.min() + 1e-6)
#         img5 = (img5 - img5.min())/(img5.max() - img5.min() + 1e-6)
#         img6 = (img6 - img6.min())/(img6.max() - img6.min() + 1e-6)
#         img7 = (img7 - img7.min())/(img7.max() - img7.min() + 1e-6)
#         img8 = (img8 - img8.min())/(img8.max() - img8.min() + 1e-6)

#         img1 = np.expand_dims(img1,axis =0)
#         img2 = np.expand_dims(img2,axis =0)
#         img3 = np.expand_dims(img3,axis =0)
#         img4 = np.expand_dims(img4,axis =0)
#         img5 = np.expand_dims(img5,axis =0)
#         img6 = np.expand_dims(img6,axis =0)
#         img7 = np.expand_dims(img7,axis =0)
#         img8 = np.expand_dims(img8,axis =0)

#         img = np.concatenate((img1,img2,img3,img4,img5,img6,img7,img8),axis=0)

#         return img

#     def randomCrop(self, img, length, width, height):
#         assert img.shape[1] >= length
#         assert img.shape[2] >= width
#         assert img.shape[3] >= height
#         x = random.randint(0, img.shape[1] - length)
#         y = random.randint(0, img.shape[2] - width)
#         z = random.randint(0, img.shape[3] - height )
#         img = img[:,x:x+length, y:y+width, z:z+height]
#         return img
#     def augment_image(self, image):
#         #if len(image.shape) == 4:
#         # k1 = np.random.uniform(image.shape[-3]/12,image.shape[-3]/4)
#         # k2 = np.random.uniform(image.shape[-2]/12,image.shape[-2]/4)
#         # k3 = np.random.uniform(image.shape[-1]/12,image.shape[-1]/4)
#         # cl = np.random.uniform(0.005,0.05)
#         # image = mc.mclahe(image, kernel_size=(1,int(k1),int(k2),int(k3)), clip_limit=cl)
#         image = self.random_rotation_3d(image, 90)
#         sigma = np.random.uniform(0.0,1.0,1)[0]
#         image = scipy.ndimage.filters.gaussian_filter(image, sigma, truncate=8)
#         return image

#     def unpickling(self, path):
#        file_return=pickle.load(open(path,'rb'))
#        return file_return

#     def random_rotation_3d(self, image, max_angle):
#         """ Randomly rotate an image by a random angle (-max_angle, max_angle).

#         Arguments:
#         max_angle: `float`. The maximum rotation angle.

#         Returns:
#         batch of rotated 3D images
#         """
#         size = image.shape
#         if bool(random.getrandbits(1)):
#             image1 = np.squeeze(image)
#             # rotate along z-axis
#             angle = random.uniform(-max_angle, max_angle)
#             image2 = scipy.ndimage.interpolation.rotate(image1, angle, mode='nearest', axes=(0, 1), reshape=False)

#             # rotate along y-axis
#             angle = random.uniform(-max_angle, max_angle)
#             image3 = scipy.ndimage.interpolation.rotate(image2, angle, mode='nearest', axes=(0, 2), reshape=False)

#             # rotate along x-axis
#             angle = random.uniform(-max_angle, max_angle)
#             image_rot = scipy.ndimage.interpolation.rotate(image3, angle, mode='nearest', axes=(1, 2), reshape=False)
#             #                print(i)
#         else:
#             image_rot = image
#         return image_rot.reshape(size)