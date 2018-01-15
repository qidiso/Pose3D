import scipy.io as spio 
import numpy as np 

annot_file = "/home/al17/dataset/mpi_inf_3dhp/S1/Seq1/annot.mat" 
annot = spio.loadmat(annot_file, squeeze_me=True, chars_as_strings=True)
# annot struct: (S1 Seq1 e.g.)
# annot keys: annot2, frames, cameras, annot3, univ_annot3
# annot2: 14 * [6416 * 56], 28 joints with 2D coord 
# frames: array([0,1,...,6415], dtype=uint16) 
# cameras: array([0...13], dtype=uint8)
# annot3: 14 * [6416 * 84], 28 joints with 3D corrd 


# visualize both 2d annotation and 3d annotation 
def visual_data(imgs, annot, calib):
    return 0 


# generat heatmap given a 3d point 
def gen_heatmap(p3d): 
    return 0 


def main(): 
    # TODO 
    return 0 

if __name__ == "__main__":
    # process 
    main() 