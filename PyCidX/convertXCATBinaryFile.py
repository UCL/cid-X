#!/usr/bin/env python3

import sys
import nibabel as nib
import numpy as np
import os
import platform
import json


def convertXCATBinaryFile( inputXCATBinFileName, 
                           outputDir, 
                           imageDimension, 
                           voxelSize, 
                           convertAtnToHU=False, 
                           outFileName=None, 
                           ctConversionAttenuationEnergy=140.0 ):
    
    # Derive the output file name
    if outFileName is None:
        if convertAtnToHU:
            outputFileName = outputDir + os.path.split(inputXCATBinFileName)[1].split('.bin')[0] + '_HU.nii.gz'        
        else:
            outputFileName = outputDir + os.path.split(inputXCATBinFileName)[1].split('.bin')[0] + '.nii.gz'        
    else:
            outputFileName = outputDir + outFileName     
        
        
    print(" ")
    print("  Converting with parameters:")
    print("  - XCAT input file:   " + inputXCATBinFileName )
    print("  - output directory:  " + outputDir )
    print("  - image dimension:   " + str(imageDimension))
    print("  - voxel size:        " + str(voxelSize))
    print("  - output file name : " + outputFileName )

    print(" ")

    if not os.path.exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)
        print("... Created output directory ...")


    print("... Starting conversion ...")

        
    with open(inputXCATBinFileName, "rb") as binFile:
        binArray = np.fromfile(binFile, dtype = np.float32 )
    
    try:
        binArray = binArray.reshape([imageDimension[2], imageDimension[1], imageDimension[0]])
        # rotate the binary array so that it corresponds to the DVF file
        binArray = np.rot90(binArray, 1, (0,2))
        binArray=np.flip(binArray, 0)

    except:
        print("Error: Cannot reformat binary array as needed. Check input image dimensions." )
        sys.exit(1)
        
    affine = np.diag( [-voxelSize[0],-voxelSize[1], voxelSize[2], 1 ] )
    
    if convertAtnToHU:
        
        # Need to get access to the attenuation coefficients
        curFilePath = os.path.dirname( os.path.abspath( __file__ ) )
        
        try:
            with open( os.path.join( curFilePath, 'atcoeff.json' ) ) as fp:
                attenuationCoeffs = json.load( fp )
        except:
            print("Error: Expecting json file with attenuation coefficients here: " )
            print(" -> " + curFilePath + "atcoeff.json" )
            sys.exit(1)

        # Find the element where for the first time the energy was exceeded
        energyIDX = np.min( np.where( np.array( attenuationCoeffs['E'] ) == ctConversionAttenuationEnergy ) )
        attenuationCoeffs['air'][energyIDX]
        
        # Coefficients required for conversion to HU
        # Need to scale by the voxel size. Also the XCAT phantom define the size in cm rather than mm
        # so need to use voxelSize[0]/10.0
        mu_air   = attenuationCoeffs[ 'air'  ][energyIDX] * voxelSize[0] / 10.0 
        mu_water = attenuationCoeffs[ 'water'][energyIDX] * voxelSize[0] / 10.0 
        binArray =  1000.0 * (binArray - mu_water) / (mu_water - mu_air)
    
    binNii = nib.Nifti1Image(binArray, affine)
    nib.save( binNii, outputFileName )
    
    print("Done.")
    return outputFileName
    
        


if __name__ == "__main__":
    
    print("Tool to convert XCAT binary image files into nifti images.")
    print("  Output: outputFileBaseName   Nifti image with the structure.")
    
    if len(sys.argv) < 2:
        print("Usage: convertXCATDVFTextFile.py pathXCATDVFFile outputDir nx ny nz dx dy dz")
        print("        - pathXCATBinFile    -> Path to the .bin file output by XCAT")
        print("        - outputDir          -> Path to which the output nifti image will be written. Gets the same name as the XCAT file")
        print("        - nx ny nz           -> Number of voxels in x, y, and z direction (optional, defaults to 512 x 512 x 151)")
        print("        - dx dy dz           -> Voxel spacing in x, y, and z direction (optional, defaults to 500/512 x 500/512 x 2)")
        print("        - convertToHU        -> If converting an attenuation map, if the conversion to HU values should be performed set this to 1 [0]")
        print(" ")
        print("        Running on "  + platform.architecture()[1] + " " + platform.architecture()[0] )
        sys.exit()
    
    xcatBinileNameIn = sys.argv[1]
    outputDir    = sys.argv[2]
    
    imageDimension = np.array( [256, 256, 161] )
    voxelSize      = np.array( [2.0, 2.0, 2.0] )
    convertAtnToHU = False
    
    if len(sys.argv) > 5:
        imageDimension[0] = float( sys.argv[3] )
        imageDimension[1] = float( sys.argv[4] )
        imageDimension[2] = float( sys.argv[5] )
    
    if len(sys.argv) > 8:
        voxelSize[0] = float(sys.argv[6])
        voxelSize[1] = float(sys.argv[7])
        voxelSize[2] = float(sys.argv[8])
    
    if len(sys.argv) > 9:
        if int(sys.argv[9]) == 1:
            convertAtnToHU = True

    convertXCATBinaryFile( xcatBinileNameIn, outputDir, imageDimension, voxelSize, convertAtnToHU )
    
