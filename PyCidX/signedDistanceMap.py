#!/usr/bin/env python3

import SimpleITK as sitk
import numpy as np
import nibabel as nib
import PyCidX.nibabelToSimpleITK as n2s


class SignedDistanceMap(object):
    '''
    Class to calculate a signed distance map from a given nibabel image. Uses simple ITK.
    '''

    def __init__( self, nibImageIn ):
        '''
        @param nibImageIn: The nibabel input image from which the signed distance map will be calulcated
        '''
        self.nibImageIn     = nibImageIn
        self.lowerThreshold = -9e10
        self.upperThreshold = 0                
        self.distMapImg     = None             # The output image
        self.postFilterBinomialRepetitions = 2 # Post-filtering iterations with binomial filter
    
    
    
    
    def run(self):
        # Convert from nibabel first
        sImg = n2s.nibabelToSimpleITK.sitkImageFromNib( self.nibImageIn )

        # Threshold the image
        thresholder = sitk.BinaryThresholdImageFilter()
        thresholder.SetLowerThreshold( self.lowerThreshold )
        thresholder.SetUpperThreshold( self.upperThreshold )
        thresholder.SetInsideValue(1)
        thresholder.SetOutsideValue(0)
        thresholdedImg = thresholder.Execute( sImg )

        # Generate the signed distance map
        distFilter = sitk.SignedDanielssonDistanceMapImageFilter()
        distFilter.SetUseImageSpacing( True )
        distMapImg = distFilter.Execute( thresholdedImg )
        
        # Add an offset such that the boundary is between voxels if checking for <0
        imgSpacing = distMapImg.GetSpacing()
        offsetVal = np.mean( imgSpacing ) / 2
        distMapImg = distMapImg - offsetVal  
        
        # Regularise the SDT in case ...
        if self.postFilterBinomialRepetitions > 0:
            distMapImg = sitk.BinomialBlur( distMapImg, self.postFilterBinomialRepetitions )
        
        # Convert back to nibabel 
        self.distMapImg = n2s.nibabelToSimpleITK.nibImageFromSITK( distMapImg )
        
    
    
        
        
if __name__ == '__main__':
    lsImageName = 'C:/debugData/workflow-improv/preprocessOut/levelSetOut.nii.gz'
    lsImg = nib.load( lsImageName )
    
    sdtMapper = SignedDistanceMap( lsImg )
    sdtMapper.run()
    nib.save( sdtMapper.distMapImg, 'C:/debugData/workflow-improv/preprocessOut/distMap.nii.gz' )
    
    