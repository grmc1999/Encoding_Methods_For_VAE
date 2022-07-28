

class Tuple_to_dict(object):

    def __init__(self,mode):
        self.new_sample={}
        self.mode=mode

    def __call__(self, sample):
        
        if self.mode=="sample":
            self.new_sample["x"]=sample
        elif self.mode=="target":
            self.new_sample["t"]=sample

        return self.new_sample