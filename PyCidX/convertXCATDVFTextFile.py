#!/usr/bin/env python3

import sys
import nibabel as nib
import numpy as np
import json
import os
import platform
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
import gzip
import io

knownInsideStructures = { 'bronchi' : 0 ,
                          'asc_large_int': 0,
                          'desc_large_int': 0,
                          'dias_lam': 0,
                          'dias_lv_4': 0,
                          'dias_pericardium': 0,
                          'dias_ram': 0,
                          'dias_rvm': 0,
                          'esophagus': 0,
                          'gall_bladder': 0,
                          'l_adrenal': 0,
                          'l_medulla': 0,
                          'l_renal_pelvis': 0,
                          'left_dia': 0,
                          'liver': 0,
                          'lkidney': 0,
                          'llung': 0,
                          'pancreas': 0,
                          'r_adrenal': 0,
                          'r_medulla': 0,
                          'r_renal_pelvis': 0,
                          'right_dia': 0,
                          'rkidney': 0,
                          'rlung': 0,
                          'small_intest': 0,
                          'spleen': 0,
                          'stomach': 0,
                          'thymus': 0,
                          'trachea': 0,
                          'trans_large_int': 0
                            }
                        
knownOutsideStructures = {  'chest_surface':1,
                            'l_thyroid':1,
                            'l_clavicle':1,
                            'rib':1,
                            'musc142':1,
                            'musc147':1,
                            'musc148':1,
                            'musc149':1,
                            'musc151':1,
                            'musc156':1,
                            'musc169':1,
                            'musc170':1,
                            'musc171':1,
                            'musc172':1,
                            'musc173':1,
                            'musc1802':1,
                            'musc1803':1,
                            'musc224':1,
                            'musc225':1,
                            'musc226':1,
                            'musc227':1,
                            'musc228':1,
                            'musc229':1,
                            'musc234':1,
                            'musc235':1,
                            'musc236':1,
                            'musc237':1,
                            'musc238':1,
                            'musc239':1,
                            'musc240':1,
                            'musc241':1,
                            'musc242':1,
                            'musc243':1,
                            'musc251':1,
                            'musc256':1,
                            'musc257':1,
                            'musc258':1,
                            'musc262':1,
                            'musc263':1,
                            'musc269':1,
                            'musc270':1,
                            'musc273':1,
                            'musc274':1,
                            'musc279':1,
                            'musc280':1,
                            'musc281':1,
                            'musc284':1,
                            'musc294':1,
                            'musc295':1,
                            'musc31':1,
                            'musc54':1,
                            'musc549':1,
                            'musc59':1,
                            'musc60':1,
                            'musc61':1,
                            'musc619':1,
                            'musc621':1,
                            'musc626':1,
                            'musc63':1,
                            'musc896':1,
                            'musc897':1,
                            'musc900':1,
                            'musc901':1,
                            'r_clavicle':1,
                            'rib_musc':1,
                            'rrib1':1,
                            'rrib10':1,
                            'rrib1c':1,
                            'rrib2':1,
                            'rrib2c':1,
                            'rrib3':1,
                            'rrib3c':1,
                            'rrib4':1,
                            'rrib4c':1,
                            'rrib5':1,
                            'rrib5c':1,
                            'rrib6':1,
                            'rrib6c':1,
                            'rrib7':1,
                            'rrib7c':1,
                            'rrib8':1,
                            'rrib8c':1,
                            'rrib9':1,
                            'rrib9c':1,
                            'lrib1':1,
                            'lrib10':1,
                            'lrib1c':1,
                            'lrib2':1,
                            'lrib2c':1,
                            'lrib3':1,
                            'lrib3c':1,
                            'lrib4':1,
                            'lrib4c':1,
                            'lrib5':1,
                            'lrib5c':1,
                            'lrib6':1,
                            'lrib6c':1,
                            'lrib7':1,
                            'lrib7c':1,
                            'lrib8':1,
                            'lrib8c':1,
                            'lrib9':1,
                            'lrib9c':1,
                            'sternum':1 }


def processDVFTextFileContents(decodedFile, tissueClassDict, labelVol, dvfVol ):
    """
    Function that processes the file contents and puts it into the variables
    :param decodedFile: The file object, either read with open or gzip and decoder.
    :param tissueClassDict: The dictionary holding the tissue classes (assumed to be empty).
    :param labelVol: The numpy array holding the tissue labels.
    :param dvfVol: The numpy array holding the dvf.
    """


    # Counter that keeps track of how many classes were found
    nextTissueClassNum = 1

    while (True):
        # Content can be array of variable length
        content = decodedFile.readlines(1)

        # Check for end of file
        if content == []:
            break

        for c in content:
            # Only analyse those content lines that hold vector information

            if c.find('vector') == -1:
                continue

            splitC = c.split()

            # Test if the current tissue class was used before if not add it to the dictionary
            try:
                curTissueClassID = tissueClassDict[splitC[0]]

            except:
                tissueClassDict[splitC[0]] = nextTissueClassNum
                curTissueClassID = nextTissueClassNum
                nextTissueClassNum += 1

            # Get the current image index
            curIDX_x = int(splitC[2])
            curIDX_y = int(splitC[3])
            curIDX_z = int(splitC[4])

            # Fill the label volume
            labelVol[curIDX_x, curIDX_y, curIDX_z] = curTissueClassID

            # Get the displacement vector
            curDisplacement_x = float(splitC[6]) - curIDX_x
            curDisplacement_y = float(splitC[7]) - curIDX_y
            curDisplacement_z = float(splitC[8]) - curIDX_z

            dvfVol[curIDX_x, curIDX_y, curIDX_z, 0, 0] = curDisplacement_x
            dvfVol[curIDX_x, curIDX_y, curIDX_z, 0, 1] = curDisplacement_y
            dvfVol[curIDX_x, curIDX_y, curIDX_z, 0, 2] = curDisplacement_z


def convertXCATDVFTextFileToNiftiImage( inputXCATDVFFileName, 
                                        structuralNiftiImageFileName, 
                                        outputDirectory, 
                                        imageDimension, 
                                        voxelSize, 
                                        generateLabelData=False, 
                                        generateSlidingInOutMaskImages=False,
                                        generateDVFNiiFile=True ):
    
    print(" ")
    print("  Converting with parameters:")
    print("  - XCAT input file:      " + inputXCATDVFFileName )
    print("  - XCAT CT-like file:    " + structuralNiftiImageFileName )
    print("  - output directory:     " + outputDirectory )
    print("  - image dimension:      " + str(imageDimension) )
    print("  - voxel size:           " + str(voxelSize) )
    print(" ")


    if not os.path.exists( outputDirectory ):
        os.makedirs( outputDirectory )
        print("  ... Created output directory ...")


    # Load the mask file and check where it is != 0
    structuralNii=nib.load( structuralNiftiImageFileName )
    
    # Extract the base name from the XCAT DVF text file and generate the output file names
    XCAT_base_file_name        = os.path.split(inputXCATDVFFileName)[1].split('.txt')[0]
    dvfImageFileName           = outputDirectory + XCAT_base_file_name + '__dvf.nii.gz'
    labelImageFileName         = outputDirectory + XCAT_base_file_name + '__label.nii.gz'
    labelDictFileName          = outputDirectory + XCAT_base_file_name + '__labelDict.json'
    labelInImageFileName       = outputDirectory + XCAT_base_file_name + '__labelIn.nii.gz'
    labelOutImageFileName      = outputDirectory + XCAT_base_file_name + '__labelOut.nii.gz'
    
    # Generate images that will hold the extracted information
    print("  ... Starting conversion ...")
    dvfVol   = np.zeros(np.hstack( (imageDimension,1,3) ), dtype = np.float32) # displacements
    # dvfVol[:] = np.nan
    labelVol = np.zeros(imageDimension, dtype = np.int16)
    
    # Dictionary holding the tissue classes 
    tissueClassDict = {}
    
    # Read in the text file and analyse at the same time
    print("  ... Reading file and analysing file contents ...")

    if inputXCATDVFFileName.endswith('.gz'):
        with gzip.open(inputXCATDVFFileName, mode='r') as f:
            with io.TextIOWrapper(f, encoding='utf-8') as decoder:
                processDVFTextFileContents( decoder, tissueClassDict, labelVol, dvfVol )
    else:
        with open(inputXCATDVFFileName, mode='r') as f:
            processDVFTextFileContents( f, tissueClassDict, labelVol, dvfVol )
    
    if generateSlidingInOutMaskImages:
        # Split label image into inside/outside
        print( "  ... Splitting label image into inside/outside structures ..." )
        
        # Purposely set to float here since they will be the basis for the level-set evolution
        # --> speed and initial image
        labelInsideVol  = np.zeros_like( labelVol, dtype=np.float32 )
        labelOutsideVol = np.zeros_like( labelVol, dtype=np.float32 )
        
        # Collect a dictionary that contains the intensities of defined structures
        tissueClassIntensityDict = {}
        maxKnownInsideMeanIntensityValue = -99999999
        # Calculate a mean rib intensity
        meanRibIntensity   = 0
        numRibClassesFound = 0
        
        meanMinLungIntensity = 0
        numLungClassesFound  = 0
        # Most superior position of 
        lungMostSupIDX = 0
        
        for curTissueClass in tissueClassDict.keys():
            
            curTissueIDXs = np.where( labelVol == tissueClassDict[ curTissueClass ] )
            
            tissueClassIntensityDict[curTissueClass] = np.array( [ np.min ( structuralNii.get_fdata()[ curTissueIDXs ] ),
                                                                   np.mean( structuralNii.get_fdata()[ curTissueIDXs ] ),
                                                                   np.max ( structuralNii.get_fdata()[ curTissueIDXs ] ) ] )
            
            # Accumulate rib and lung intensities for averaging
            if curTissueClass.find('rib') != -1:
                meanRibIntensity   += tissueClassIntensityDict[curTissueClass][1]
                numRibClassesFound += 1

            if curTissueClass.find('lung') != -1:
                meanMinLungIntensity   += tissueClassIntensityDict[curTissueClass][0]
                numLungClassesFound += 1
                lungMostSupIDX = np.max( [ np.max( curTissueIDXs[2] ), lungMostSupIDX ] )
            
            # find if it is an inside label
            for insideStructName in knownInsideStructures.keys():
                if curTissueClass.endswith( insideStructName ):
                    labelInsideVol[ curTissueIDXs ] = 1
                    maxKnownInsideMeanIntensityValue = np.max( [tissueClassIntensityDict[curTissueClass][1], maxKnownInsideMeanIntensityValue])
    
            # find if it is an outside label
            for outsideStructName in knownOutsideStructures.keys():
                if curTissueClass.endswith( outsideStructName ):
                    labelOutsideVol[ curTissueIDXs ]  = 1
            del curTissueIDXs
        
        meanRibIntensity  = meanRibIntensity  / numRibClassesFound
        meanMinLungIntensity = meanMinLungIntensity / numLungClassesFound
        
        print( "  ... Adding thresholded structures from image (e.g. lung and spine) ..." )
        boneThreshold = 0.5 * ( meanRibIntensity + maxKnownInsideMeanIntensityValue)
        labelOutsideVol[ np.where(structuralNii.get_fdata() >= boneThreshold ) ]=1;
        
        lungLowerThreshold = meanMinLungIntensity * 0.98
        lungUpperThreshold = meanMinLungIntensity * 1.02
        
        lungVol = np.zeros_like( structuralNii.get_fdata() )
        
        lungVol[np.where( (structuralNii.get_fdata() >=lungLowerThreshold) & (structuralNii.get_fdata() <=lungUpperThreshold ))] = 1
        lungVol = binary_erosion(lungVol).astype(lungVol.dtype)
        
        # Allow 10 mm above most superior index 
        numVoxAboveMostSupLungPos = int(np.ceil(10.0/voxelSize[2]))
        
        try:
            lungVol[:,:,lungMostSupIDX+numVoxAboveMostSupLungPos:] = 0
        except:
            pass
        
        labelInsideVol[ np.where(lungVol !=0 ) ]=1

        print( "     --> bone threshold:       " + str( boneThreshold ) )
        print( "     --> lung lower threshold: " + str( lungLowerThreshold ) )
        print( "     --> lung upper threshold: " + str( lungUpperThreshold ) )
        
    
    
    # Generate the nifti output images
    print( "  ... Saving output files ..." )
    affine = np.diag( [-voxelSize[0],-voxelSize[1], voxelSize[2], 1 ] )

    if generateDVFNiiFile:
        # Calculate a mask image from the structural one and find the region outside which needs to be replaced with the closest inside values
        maskData = np.ones_like( structuralNii.get_fdata() )
        maskData[ np.where(structuralNii.get_fdata() > np.min(structuralNii.get_fdata()[structuralNii.get_fdata()!=0])) ] = 0
        
        maskData = binary_dilation(maskData).astype(maskData.dtype)
        indices = distance_transform_edt( maskData, voxelSize, return_distances=False, return_indices=True )

        # DVF was in voxels, thus need to convert to mm
        dvfVol[ :, :, :, 0, 0 ] = dvfVol[ :, :, :, 0, 0 ] * np.abs(voxelSize[0]) * (-1)
        dvfVol[ :, :, :, 0, 1 ] = dvfVol[ :, :, :, 0, 1 ] * np.abs(voxelSize[1]) * (-1)
        dvfVol[ :, :, :, 0, 2 ] = dvfVol[ :, :, :, 0, 2 ] * np.abs(voxelSize[2])
        
        print( "  ... approximating DVF outside phantom ..." )
        # Correct the outside region
        dvfVol[:,:,:,0,0]=dvfVol[:,:,:,0,0][ indices[0,:,:,:],indices[1,:,:,:],indices[2,:,:,:] ]
        dvfVol[:,:,:,0,1]=dvfVol[:,:,:,0,1][ indices[0,:,:,:],indices[1,:,:,:],indices[2,:,:,:] ]
        dvfVol[:,:,:,0,2]=dvfVol[:,:,:,0,2][ indices[0,:,:,:],indices[1,:,:,:],indices[2,:,:,:] ]
        
        print( "  --> Writing DVF image to:   " + dvfImageFileName )
        dvfNii = nib.Nifti1Image( dvfVol, affine )
        
        # Set the correct header parameters for transformations with niftyReg
        try:
            dvfNii.header['intent_code'] = 1007
            dvfNii.header['intent_p1']   = 1 
            dvfNii.header['intent_name'] ='NREG_TRANS' 
            dvfNii.header['sform_code'] = 1
        except:
            dvfNii.get_header()['intent_code'] = 1007
            dvfNii.get_header()['intent_p1']   = 1 
            dvfNii.get_header()['intent_name'] ='NREG_TRANS' 
            dvfNii.get_header()['sform_code'] = 1
        nib.save( dvfNii, dvfImageFileName )
        del dvfNii
    else:
        # Clear the image file name that was not generated
        dvfImageFileName = ''
    del dvfVol
    
    if generateLabelData:
        print( "  --> Writing label image to: " + labelImageFileName )
        labelNii = nib.Nifti1Image(labelVol, affine)
        nib.save( labelNii, labelImageFileName )
        del labelNii
    else:
        # Clear the image file name that was not generated
        labelImageFileName = ''
    
    del labelVol
    
    if generateSlidingInOutMaskImages:
        # Note: Using label values that can go directly into the levelset evolution
        print( "  --> Writing outside label image to: " + labelOutImageFileName )
        labelOutNii = nib.Nifti1Image( 1 - labelOutsideVol, affine )
        nib.save( labelOutNii, labelOutImageFileName )
        del labelOutNii, labelOutsideVol
        
        print( "  --> Writing inside label image to: " + labelInImageFileName )
        labelInNii = nib.Nifti1Image( 1 - 2*labelInsideVol, affine)
        nib.save( labelInNii, labelInImageFileName )
        del labelInNii, labelInsideVol
    else:
        # Clear the image file name that was not generated
        labelInImageFileName  = ''
        labelOutImageFileName = ''


    if generateLabelData:
        print( "  --> Writing label dict to:  " + labelDictFileName )
        with open( labelDictFileName, 'w' ) as fp:
            try: 
                json.dump(tissueClassDict, fp, indent = '  ', sort_keys=True)
            except:
                json.dump(tissueClassDict, fp, indent = 2, sort_keys=True)
    else:
        labelDictFileName = '' 
    
    
    outputFileNames = {  'dvfImageFileName'      : dvfImageFileName,
                         'labelImageFileName'    : labelImageFileName,
                         'labelDictFileName'     : labelDictFileName,
                         'labelInImageFileName'  : labelInImageFileName,
                         'labelOutImageFileName' : labelOutImageFileName }

    print( "Done." )
    
    return outputFileNames
    



if __name__ == "__main__":
    
    print("Tool to convert XCAT DVF text output files into nifti images.")
    # The commented files will be written if the boolean variables below are set accordingly (generateLabels/generateLSInputData)
    # print("  Output: outputDir/XCAT_outFileName__label.nii.gz     label image with an index for each anatomical structure.")
    # print("  Output: outputDir/XCAT_outFileName__labelDict.json   dictionary of the label number - to label name association.")
    print("  Output: outputDir/XCAT_outFileName__dvf.nii.gz       DVF image with a displacement vector (in voxels).")
    
    if len(sys.argv) < 2:
        print("Usage: convertXCATDVFTextFile.py pathXCATDVFFile pathToNiftiImage outputDir nx ny nz dx dy dz")
        print("        - pathXCATDVFFile       -> Path to the DVF text file output by XCAT")
        print("        - pathToNiftiImage      -> Path to an image showing the XCAT anatomy in nifti image format")
        print("        - outputDir             -> Path to where the output will be saved ")
        print("        - nx ny nz              -> Number of voxels in x, y, and z direction (optional, defaults to 256 x 256 x 161)")
        print("        - dx dy dz              -> Voxel spacing in x, y, and z direction (optional, defaults to 2 x 2 x 2)")
        print("        - topZ0                 -> Number of z-slices (from superior) in the lung-segmentation that is forced to zero, optional.")
        print(" ")
        print("        Running on "  + platform.architecture()[1] + " " + platform.architecture()[0] )
        sys.exit()
    
    XCATdvfTextFileNameIn              = sys.argv[1]
    XCATstructuralNiftiImageFileNameIn = sys.argv[2]
    
    outputDirectoryName = sys.argv[3]
    
    # Default values
    imageDimension = np.array( [256,256,161] )
    voxelSize      = np.array( [2.0, 2.0, 2.0] )
    clipLungThresholdTop = None
    
    if len(sys.argv) > 6:
        imageDimension[0] = float( sys.argv[4] )
        imageDimension[1] = float( sys.argv[5] )
        imageDimension[2] = float( sys.argv[6] )
    
    if len(sys.argv) > 9:
        voxelSize[0] = float( sys.argv[7] )
        voxelSize[1] = float( sys.argv[8] )
        voxelSize[2] = float( sys.argv[9] )
    
    if len(sys.argv) > 10:
        clipLungThresholdTop = int( sys.argv[10] )

    generateLabels      = False
    generateLSInputData = False
    generateDVFNiiFile  = True
    
    convertXCATDVFTextFileToNiftiImage(XCATdvfTextFileNameIn,
                                       XCATstructuralNiftiImageFileNameIn,
                                       outputDirectoryName,
                                       imageDimension,
                                       voxelSize,
                                       generateLabels,
                                       generateLSInputData,
                                       generateDVFNiiFile)

