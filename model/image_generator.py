import torch
import torchvision
import numpy as np
from PIL import Image
import pandas as pd
import pickle
import sys
import os
sys.path.append('./model')

PRESENT_MODE = False
PRESENT_IMAGES = [191, 80, 5, 79, 26, 0]

class ImageGenerator():
    def __init__(self):

        # load model
        self.model = torch.load('./model/precomputed/model.pckl', map_location='cpu')
        self.model = self.model.to('cpu')
        self.model.eval()
        self.latent_dim = self.model.latent_space

        # check how many pretreained images are present
        self.fnames = sorted(os.listdir('./example_images'))
        self.n = len(self.fnames)

        # load prelabaled data
        self.labels = pickle.load(open('./model/precomputed/labels.pickl', 'rb'))
        self.labels = self.labels[:self.n]

        # load trained feature vectors
        self.feature_vectors = pickle.load(open('./model/precomputed/feature_vectors.pickl', 'rb'))

        # set up helper functions
        self.to_img = torchvision.transforms.ToPILImage()
        self.to_tensor = torchvision.transforms.ToTensor()

        # features considered in this programm
        self.features = ["Attractive", "Bald", "Black_Hair", "Blond_Hair", "Brown_Hair", "Eyeglasses", "Gray_Hair", "Heavy_Makeup", "Male", "Mustache", "No_Beard", "Pale_Skin", "Smiling"]

        # set yup index for cyling through own images
        self.img_index_own = 0
        self.img_index_present = 0

        # load random image
        self.random()

    def random(self):
        # pick random image
        if not PRESENT_MODE:  
            self.img_idx = np.random.randint(self.n)
        else:
            self.img_idx = PRESENT_IMAGES[self.img_index_present]
            self.img_index_present += 1
            if self.img_index_present >= len(PRESENT_IMAGES):
                self.img_index_present = 0
        self.img_fname = self.fnames[self.img_idx]
        self.img_tensor = self.to_tensor(Image.open('./example_images/' + self.img_fname))

        # set up deafults
        defaults = {}
        for feature in self.features:
            defaults[feature] = .5 if self.labels[feature][self.img_idx] == 1 else -.5
        self.defaults = defaults
        self.settings = defaults.copy()

        print("showing image with id {}".format(self.img_idx))
        return

    def random_own(self):
        if not os.path.isdir('./additional_images_preprocessed/'):
            return False

        own_fnames = os.listdir('./additional_images_preprocessed/')
        n = len(own_fnames)
        if n == 0:
            return False

        self.img_index_own += 1
        self.img_index_own = self.img_index_own % n
        self.img_fname = own_fnames[self.img_index_own]
        self.img_tensor = self.to_tensor(Image.open('./additional_images_preprocessed/' + self.img_fname))
    
        # set up deafults
        defaults = {}
        for feature in self.features:
            defaults[feature] = 0.
        self.defaults = defaults
        self.settings = defaults.copy()
        return True


    def get_defaults(self):
        return self.defaults

    def set_settings(self, attractive, bald, black_hair, blond_hair, brown_hair, glasses, gray_hair, makeup, male, mustache, no_beard, pale, smiling):
        settings = {}
        settings["Attractive"] = attractive
        settings["Bald"] = bald
        settings["Black_Hair"] = black_hair
        settings["Blond_Hair"] = blond_hair
        settings["Brown_Hair"] = brown_hair
        settings["Eyeglasses"] = glasses
        settings["Gray_Hair"] = gray_hair
        settings["Heavy_Makeup"] = makeup
        settings["Male"] = male
        settings["Mustache"] = mustache
        settings["No_Beard"] = no_beard
        settings["Pale_Skin"] = pale
        settings["Smiling"] = smiling
        self.settings = settings
        return

    def render(self):
        # calc feature vect additive
        additive = np.zeros((self.latent_dim))
        for feature in self.features:
            additive += self.feature_vectors[feature] * (self.settings[feature]- self.defaults[feature])
        additive = torch.Tensor(additive)

        # run model
        with torch.no_grad():
            x = self.img_tensor.to('cpu').view(-1,3,64,64)
            mu, logvar = self.model.encode(x)
            mu += additive.to('cpu')
            x_recon = self.model.decode(mu)
            x_recon = x_recon.cpu().view(3,64,64)


        img = self.to_img(x_recon)
        img = img.resize((64*6, 64*6), Image.NEAREST)
        self.img = img
        return img

    def save(self, fname, size=3):
        self.img.resize((64*size, 64*size), Image.NEAREST).save(fname)
        return
