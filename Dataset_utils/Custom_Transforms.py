

class Tuple_to_dict(object):

    def __init__(self):
        self.new_sample={}
        self.flag=0

    def __call__(self, sample):
        
        if self.flag==0:
            self.new_sample["x"]=sample
            self.flag=self.flag+1
        else:
            self.new_sample["y"]=sample
            self.flag=0

        return self.new_sample