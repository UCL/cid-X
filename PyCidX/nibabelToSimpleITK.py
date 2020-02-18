#!/usr/bin/env python3 

import SimpleITK as sITK
import nibabel as nib
import numpy as np


class nibabelToSimpleITK(object):

    
    def __init__( self, nibabelImageIn ):
        self.nibImgIn = nibabelImageIn
        self.sitkImg  = None
        
        if (self.nibImgIn.header['dim'][0] != 3):
            print("WARNING: This class is currently only intended for 3D images")
        
        
        
        
    def convertToSITK(self):
        # Generate the image
        # Note that the order of the axes is reverted...
        self.sitkImg = sITK.GetImageFromArray( np.transpose( self.nibImgIn.get_data(), [2,1,0] ) )
        
        # Set the image geometry
        # - origin
        self.sitkImg.SetOrigin( self.nibImgIn.affine[:3,3] * np.array([-1, -1, 1]) )

        # - spacing
        self.sitkImg.SetSpacing( self.nibImgIn.header['pixdim'][1:4].astype( np.double ) )
        
        # - direction
        dirMatrix = self.nibImgIn.affine[:3,:3].copy()
        dirMatrix[:,0] = dirMatrix[:,0] / np.linalg.norm( dirMatrix[:,0] ) * (-1) 
        dirMatrix[:,1] = dirMatrix[:,1] / np.linalg.norm( dirMatrix[:,1] ) * (-1) 
        dirMatrix[:,2] = dirMatrix[:,2] / np.linalg.norm( dirMatrix[:,2] ) 
        self.sitkImg.SetDirection( dirMatrix.transpose().reshape(-1) )
        
        return self.sitkImg

        
    
    def pushSITKImageContentIntoOriginalNibabelData( self, sitkImgIn ):
        
        # Extract the image data from the SimpleITK image object
        recoveredImgArray =  sITK.GetArrayFromImage( sitkImgIn )
        
        # Revert the re-ordering of the axes
        recoveredImgArray = np.transpose(recoveredImgArray, [2,1,0])
        
        # Generate a nifti image
        outNibImg = nib.Nifti1Image( recoveredImgArray , 
                                     self.nibImgIn.affine, 
                                     self.nibImgIn.header )
        
        return outNibImg


if __name__ == '__main__':
    
    # Load the test image
    nImg = nib.load( 'D:/debugData/test_01_MCR.nii.gz' )

    # Generate the converter and perform the conversion
    converter = nibabelToSimpleITK( nImg )
    convertedSITKImg = converter.convertToSITK()
    
    # Revert the changes and save the image
    recoveredNibImg = converter.pushSITKImageContentIntoOriginalNibabelData(convertedSITKImg)
    nib.save( recoveredNibImg, 'D:/debugData/test_01_MCR_recovered.nii.gz' )
    
