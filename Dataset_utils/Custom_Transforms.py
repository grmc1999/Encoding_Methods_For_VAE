

class Tuple_to_dict(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        new_sample={}
        new_sample["x"]=sample[0]
        new_sample["y"]=sample[1]

        return new_sample