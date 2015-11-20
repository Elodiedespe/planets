
# System import 
import numpy
import os
import subprocess
import scipy.signal
import glob
import shutil
import pandas as pd
import sys


# Plot import
import matplotlib.pyplot as plt

# IO import
import nibabel


def to_id(in_file):
    """ Convert file to itself (get .minf file).
    """
    cmd = ["AimsFileConvert", "-i", in_file, "-o", in_file]
    print "Executing: '{0}'.".format(" ".join(cmd))
    subprocess.check_call(cmd)
    return in_file


def reorient_image(input_axes, in_file, output_dir):
    """ Rectify the orientation of an image.
    """
    reoriented_file = in_file.split('.')[0] + "_reorient.nii.gz"
    reoriented_file = os.path.join(output_dir, os.path.basename(reoriented_file))
    cmd = ["AimsFlip", "-i", in_file, "-o", reoriented_file, "-m", input_axes]
    print "Executing: '{0}'.".format(" ".join(cmd))
    subprocess.check_call(cmd)
    return reoriented_file

    
def mri_to_ct(t1_nii, ct_nii, min_thr, output_dir, flip, verbose= 0):
    
    # Output autocompletion
    ct_modify_nii = os.path.join(output_dir, "ct_modify.nii.gz")
    ct_brain_nii = os.path.join(output_dir, "ct_cut_brain.nii.gz")
    register_t1_nii = os.path.join(output_dir, "t1_to_cut_ct.nii.gz")
    transformation = os.path.join(output_dir, "t1_to_cut_ct.txt")
    t1_ct_nii = os.path.join(output_dir, "t1_to_ct.nii.gz")
    ct_cut_reoriented_nii = os.path.join(output_dir, "ct_cut_reoriented.nii.gz")
    cut_brain_index_fileName = os.path.join(output_dir, "ct_brain_index.txt")
    
    # Load ct and modify the data for brain extraction
    print 'ct_nii: ', ct_nii
    ct_im = nibabel.load(ct_nii)
    print "ok load"
    ct_data = ct_im.get_data()
    ct_shape = ct_data.shape
    ct_data[numpy.where(ct_data < 0)] = 0
    nibabel.save(ct_im, ct_modify_nii)
    # Detect the neck
    ct_im = nibabel.load(ct_modify_nii)
    ct_data = ct_im.get_data()
    power = numpy.sum(numpy.sum(ct_data, axis=0), axis=0)
    powerfilter = scipy.signal.savgol_filter(power, window_length=11, polyorder=1)
    mins = (numpy.diff(numpy.sign(numpy.diff(powerfilter))) > 0).nonzero()[0] + 1
    global_min = numpy.inf
    global_min_index = -1
    for index in mins:
        if powerfilter[index] > min_thr and global_min > powerfilter[index]:
            global_min = powerfilter[index]
            global_min_index = index
    cut_brain_index_file = open(cut_brain_index_fileName, "w")
    cut_brain_index_file.write(str(global_min_index))
    cut_brain_index_file.close()

    # Diplay if verbose mode
    if verbose == 1:
        x = range(power.shape[0])
        plt.plot(x, power, '.', linewidth=1)
        plt.plot(x, powerfilter, '--', linewidth=1)    
        plt.plot(x[global_min_index], powerfilter[global_min_index], "o")       
        plt.show()
    # Cut the image
    ct_cut_data = ct_data[:, :, range(global_min_index, ct_data.shape[2])]
    brain_im = nibabel.Nifti1Image(ct_cut_data, ct_im.get_affine())
    nibabel.save(brain_im, ct_brain_nii)
    # Reorient ct brain image
    if flip == 1:
        print "flip OK"
        ct_cut_reoriented_nii = reorient_image("XXYY", ct_brain_nii, output_dir)
    else:
        print "NO flip"
        ct_cut_reoriented_nii = to_id(ct_brain_nii)
    # Register
    cmd = ["flirt", "-cost", "normmi", "-omat", transformation, "-in", t1_nii,
           "-ref", ct_cut_reoriented_nii, "-out", register_t1_nii]
    print "Executing: '{0}'.".format(" ".join(cmd))
    subprocess.check_call(cmd)
    
    # Send the t1 to the ct original space
    #t1_data = nibabel.load(register_t1_nii).get_data()
    #ct_t1_data = numpy.zeros(ct_data.shape)
    #ct_t1_data[:, :, range(global_min_index, ct_data.shape[2])] = t1_data
    #t1_ct_im = nibabel.Nifti1Image(ct_t1_data, ct_im.get_affine())
    #nibabel.save(t1_ct_im, t1_ct_nii)

    # Reorient the t1
    if flip == 1:
        t1_ct_reoriented_nii = reorient_image("XXYY", t1_ct_nii, output_dir)
    else:
        t1_ct_reoriented_nii = to_id(t1_ct_nii)
    return transformation, ct_cut_reoriented_nii, global_min_index

    
def labels_to_ct( t1_nii, ct_cut_reoriented_nii, transformation, labels_nii, transform_warped, global_min_index, output_dir):
    """ Register the labels to the CT.
    """

    # Output autocompletion
    registered_labels_nii = os.path.join(output_dir, "labels_to_cut_ct.nii.gz")
    labels_ct_nii = os.path.join(output_dir, "labels_to_ct.nii.gz")
    convert_trans_itk = os.path.join(output_dir, "convert_trans_itk.txt")
    label_native_ct_nii = os.path.join(output_dir, "label_native_ct.nii")

    # Convert affine transformation from fsl2Ras (ants) 
    cmd = "c3d_affine_tool " + \
          " -ref " + ct_cut_reoriented_nii + \
          " -src " + t1_nii + " " + transformation + \
          " -fsl2ras " + \
          " -oitk " + convert_trans_itk
    print "Executing: " + cmd
    os.system(cmd)

    # Apply the affine and warp transformation
    print labels_nii
    print registered_labels_nii
    print ct_cut_reoriented_nii
    print transform_warped
    print convert_trans_itk
    cmd = "antsApplyTransforms " + \
          " --float " + \
          " --default-value 0 " + \
          " --input %s " % (labels_nii) + \
          " --input-image-type 3 " + \
          " --interpolation NearestNeighbor " + \
          " --output %s " % (registered_labels_nii) + \
          " --reference-image %s " % (ct_cut_reoriented_nii) + \
          " --transform [ %s , 0] [%s , 0] " % (convert_trans_itk, transform_warped)
    print "executing: " + cmd
    os.system(cmd)
    
     # Combine transformation
    #print "ct file : ", ct_brain_nii
    #cmd = "convertwarp " + \
          #" --ref=" + ct_brain_nii + \
          #" --postmat=" + transformation + \
          #" --warp1=" + transform_warped + \
          #" --out=" + combined_trans
    #print "Executing: " + cmd
    #os.system(cmd)
    
    # Warp the labels to the ct
    #cmd = ["applywarp", "-i", labels_nii, "-o", registered_labels_nii,
           #"-r", ct_brain_nii, "-w", combined_trans, "--interp=nn"]
    #print "Executing: '{0}'.".format(" ".join(cmd))
    #subprocess.check_call(cmd)
    
    if flip == 1:
        labels_ct_nii = reorient_image("XXYY", registered_labels_nii, output_dir)
    else:
        labels_ct_nii = to_id(registered_labels_nii)
        
    # Send the label to the native ct
    
    # Load the image
    ct_data = nibabel.load(ct_nii).get_data()
    labels_data = nibabel.load(labels_ct_nii).get_data()

    #Create empty matrix of the size of the native ct and fill it with labels_ct__nii matrix value 
    label_native_ct = numpy.zeros(ct_data.shape)
    label_native_ct[:, :, :global_min_index] = 0
    label_native_ct[:,:, global_min_index:] = labels_data

    #Create a Nifti object and save it
    labels_to_native_ct_im = nibabel.Nifti1Image(label_native_ct, ct_data.get_affine())
    nibabel.save(label_native_ct_im, label_native_ct_nii)
    
    return labels_ct_nii, label_native_ct_nii


def inverse_affine(affine):
    """ Invert an affine transformation.
    """
    invr = numpy.linalg.inv(affine[:3, :3])
    inv_affine = numpy.zeros((4, 4))
    inv_affine[3, 3] = 1
    inv_affine[:3, :3] = invr
    inv_affine[:3, 3] =  - numpy.dot(invr, affine[:3, 3])
    return inv_affine


def threed_dot(matrice, vector):
    """ Dot product between a 3d matrix and an image of 3d vectors.
    """
    res = numpy.zeros(vector.shape)
    for i in range(3):
	    res[..., i] = (matrice[i, 0] * vector[..., 0] + 
                       matrice[i, 1] * vector[..., 1] + 
                       matrice[i, 2] * vector[..., 2] +
                       matrice[i, 3])
    return res

def ct_to_rd(ct_nii, rd_nii, correct_cta, output_dir):
    """ Register the rd to the ct space.
    """
    
    # Output autocompletion
    ct_rescale_file = os.path.join(output_dir, "ct_rescale.nii.gz")

    # Load images
    ct_im = nibabel.load(ct_nii)
    ct_data = ct_im.get_data()
    rd_im = nibabel.load(rd_nii)
    rd_data = rd_im.get_data()
    cta = ct_im.get_affine()
    rda = rd_im.get_affine()

    # Correct the rda affine matrix
    
    cta[2, 2] = correct_cta
    #rda[2, 2] = 3

    # Inverse affine transformation
    icta = inverse_affine(cta)
    t = numpy.dot(icta, rda)

    # Matricial dot product
    ct_rescale = numpy.zeros(rd_data.shape)
    dot_image = numpy.zeros(rd_data.shape + (3, ))
    x = numpy.linspace(0, rd_data.shape[0] - 1, rd_data.shape[0])
    y = numpy.linspace(0, rd_data.shape[1] - 1, rd_data.shape[1])
    z = numpy.linspace(0, rd_data.shape[2] - 1, rd_data.shape[2])
    xg, yg, zg = numpy.meshgrid(y, x, z)
    print 'dot image shape: ', dot_image.shape
    print 'yg shape: ', yg.shape
    print 'xg shape: ', xg.shape
    print 'zg shape: ', zg.shape
    dot_image[..., 0] = yg
    dot_image[..., 1] = xg
    dot_image[..., 2] = zg
    dot_image = threed_dot(t, dot_image)

    cnt = 0
    print rd_data.size
    for x in range(rd_data.shape[0]):
        for y in range(rd_data.shape[1]):
            for z in range(rd_data.shape[2]):
            #for z in range(cut_brain_index, rd_data.shape[2]):
                if cnt % 100000 == 0:
                    print cnt  
                cnt += 1          
                voxel_ct = dot_image[x, y, z]
                if (voxel_ct > 0).all() and (voxel_ct < (numpy.asarray(ct_data.shape) - 1)).all():
                    ct_voxel = numpy.round(voxel_ct)
                    ct_rescale[x, y, z] = ct_data[ct_voxel[0], ct_voxel[1], ct_voxel[2]]

    ct_rescale_im = nibabel.Nifti1Image(ct_rescale, rda)
    nibabel.save(ct_rescale_im, ct_rescale_file)
    
  
    

    return ct_rescale_file

def labels_to_rd(rd_nii, labels_ct_nii, output_dir):
   
    """ Register the labels to the rd.
    """
    # Output autocompletion
    labels_rd_nii = os.path.join(output_dir, "labels_to_rd.nii.gz")
    
    rd_im = nibabel.load(rd_nii)
    rd_data = rd_im.get_data()
    labels_data = nibabel.load(labels_ct_nii).get_data()
    rd_labels_data = numpy.zeros(rd_data.shape)
    rd_labels_data[:, :, -labels_data.shape[2]:] = labels_data
    labels_rd_im = nibabel.Nifti1Image(rd_labels_data, rd_im.get_affine())
    nibabel.save(labels_rd_im, labels_rd_nii)

    return labels_rd_nii

    
if __name__ == "__main__":


    # Global parameters
    BASE_PATH = "/neurospin/grip/protocols/MRI/dosimetry_elodie_2015/clemence"
    ATLAS_PATH = "/neurospin/grip/protocols/MRI/dosimetry_elodie_2015/clemence/atlas"
    nii_path = os.path.join(BASE_PATH, "sujet_18_rt")
    output_path = os.path.join(BASE_PATH, "results_from_label_to_rd_after_ants")
    subjects_csv = os.path.join(nii_path, "clinical_data.csv")
    
    # get the appropriate labels
    
    labels_2 = os.path.join(ATLAS_PATH, "atlas_0-2/ANTS2-0Years_brain_ANTS_LPBA40_atlas.nii.gz")
    labels_5 = os.path.join(ATLAS_PATH, "atlas_5_9/ANTS9-5Years3T_brain_ANTS_LPBA40_atlas.nii.gz")
    labels_2_5 = os.path.join(ATLAS_PATH, "atlas_2_5/ANTS2-5Years_brain_LPBA40_atlas.nii.gz")


    # Keep the valid subject
    valid_subject_dirs = [os.path.join(nii_path, dir_name)
                      for dir_name in os.listdir(nii_path)
                      if os.path.isdir(os.path.join(nii_path, dir_name))]
    
    #valid_subject_dirs.sort()
    
    # Read the dataframe clinical data: age, orientation of the ct
    df_subjects = pd.read_csv(subjects_csv)  

                         

    # Go through all subjects
    for subject_path in valid_subject_dirs[15:17]:
        #subject_path = os.path.join(nii_path, 'sujet_024_VM')
        print "Processing: '{0}'...".format(subject_path)
        
        # Get subject id
        if not nii_path.endswith(os.path.sep):
            nii_path = nii_path + os.path.sep
        #subj_id = subject_path.replace(nii_path, "").split(os.path.sep)[0]
        subj_id = os.path.basename(subject_path)
        print subj_id
        # Select the correct atlas according to age
        subj_age = df_subjects.ART[df_subjects.anonym_nom == subj_id].values[0]
        print subj_age
        
        if subj_age < 2:
            template_labels = labels_2
            print " under 2 years old"
        elif subj_age > 5:
            template_labels = labels_5
            print " over 5 years old"
        else:
            template_labels = labels_2_5
            print " between 2 and 5 years old"
        # Create output directory and skip processing if already
        output_dir = os.path.join(output_path, subj_id)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        #Check whether ct image has to be flipped
        flip = df_subjects.check_flip[df_subjects.anonym_nom == subj_id].values[0]
        
        # Get the t1, the inv_trans (from atlas to t1), the rd and the ct of the patient
        t1_nii = glob.glob(os.path.join(subject_path, "mri", "*", subj_id + '_T1.nii'))[0]
        ct_nii = glob.glob(os.path.join(subject_path, "ct", "*.nii.gz"))[0]
        rd_nii = glob.glob(os.path.join(subject_path, "rd", "*.nii"))[0]
        transform_warped = os.path.join(subject_path, "transform_warped", "transformInverseComposite.h5")
        
        print "Executing: %s" % (t1_nii)
        print "Executing: %s" % (ct_nii)
        print "Executing: %s" % (rd_nii)

        print "Executing: %s" % (transform_warped)
        
        """
        if not os.path.isfile(os.path.join(output_dir, "ct_cut_brain.nii.gz")):
            transformation, ct_cut_reoriented_nii, cut_brain_index = mri_to_ct(t1_nii, ct_nii, 50000, output_dir, flip, verbose=0)
        else:
            print "MRI to ct transformation already processed"

        if not os.path.isfile(os.path.join(output_dir, "labels_to_ct.nii.gz")):
            labels_ct_nii = labels_to_ct(t1_nii, ct_cut_reoriented_nii, transformation, template_labels, transform_warped, global_min_index,output_dir)
        else:
            print "Labels to ct trabsformation already processed"
        """
        ct_cut_brain = os.path.join(output_dir, 'ct_cut_brain.nii.gz')
        cut_brain_index_fileName = os.path.join(output_dir, "ct_brain_index.txt")
        cut_brain_index_file = open(cut_brain_index_fileName, "r")
        cut_brain_index = int(cut_brain_index_file.read())
        cut_brain_index_file.close()
        
        # Correct the Rzz in the ct affine to find the correct correspondance in the physical coordonnates
        correct = {'sujet_005_BZ': 1.21, 'sujet_007_MM': 1.21,
                   'sujet_010_SA': 1.21, 'sujet_011_PA': 1,
                   'sujet_014_WS': 1, 'sujet_015_MI': 1.21,
                   'sujet_016_DG': 0.997, 'sujet_017_AG': 1.396,
                   'sujet_022_SK': 1, 'sujet_024_VM': 1.334,
                   'sujet_027_BL': 1.292, 'sujet_028_CH': 1,
                   'sujet_029_CT': 1.21,
                   'sujet_032_HB': 1.053, 'sujet_033_HL': 1.174,
                   'sujet_034_HI': 1, 'sujet_038_ZH': 1}

        if not os.path.isfile(os.path.join(output_dir, "ct_rescale.nii.gz")):
            ct_rescale_file = ct_to_rd(ct_cut_brain, rd_nii, correct[subj_id], output_dir)
        else:
            print "ct to rd transformation already processed"

        """
        if not os.path.isfile(os.path.join(output_dir, "labels_to_rd.nii.gz")):
            labels_rd_nii = labels_to_rd(rd_nii, labels_ct_nii, output_dir)
        else:
            print "labels to rd transformation already processed"
        """
        
       
        

