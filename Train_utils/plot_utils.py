import numpy as np
import matplotlib.pyplot as plt
import os

class plot_training_sample(object):
    def __init__(self,images_idxs,titles=None,image_save_dir=None):
        self.images_idxs=images_idxs
        self.titles=titles
        self.image_save_dir=image_save_dir

        #plt.ion()
        self.figure,self.axs=plt.subplots(1,len(self.images_idxs),figsize=(10,5),constrained_layout=True)

        
    def tensor_image_to_image(self,image):
        return (np.swapaxes(np.swapaxes(image,0,1),1,2)*255).astype(np.uint8)

    def plot(self,outs,epoch=None):

        for idx in range(len(self.images_idxs)):
            self.axs[idx].clear()
            image=((outs[self.images_idxs[idx]]).cpu().detach().numpy())[0]
            if len(image.shape)==3:
                image=self.tensor_image_to_image(image)
            self.axs[idx].imshow(image)
            if self.titles!=None:
                self.axs[idx].set_title(self.titles[idx])

        #self.figure.canvas.draw()
        self.figure.show()
        if epoch!=None:
            plt.savefig(os.path.join(self.image_save_dir,(str(epoch)+".png")))