#!/usr/bin/env python3

# My imports
import nibabelToSimpleITK as n2sConv
import SimpleITK as sitk
import nibabel as nib



class LevelSetEvolution(object):

    def __init__( self, initialNibabelImage, speedNibabelImage ):
        
        # The images
        # Only keep the converter objects
        self.n2sInitialImg = n2sConv.nibabelToSimpleITK( initialNibabelImage )
        self.n2sSpeedlImg  = n2sConv.nibabelToSimpleITK( speedNibabelImage   )

        # Default parameters
        self.maxRMSError        = 0.0005
        self.propagationScaling = 1.0
        self.curvatureScaling   = 20.0
        self.numberOfIterations = 5000
        
        self.paddingZ = 5
        
        # The output - set to non such that it is not written without being calculated...
        self.outLSImage = None

    
    
    
    def run(self):
        # Call the pre-processing of the images (i.e. padding)
        #self._preProcessImages()
        
                
        ''' Pre-process: Convert images to 
            1) simpleITK images
            2) float32 and 
            3) pad them with the zero-flux padding filter in the lower and upper z-direction
        '''
        
        self.n2sInitialImg.convertToSITK()
        self.n2sSpeedlImg.convertToSITK()
        
        caster= sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkFloat32)
        
        # local SimpleITK version of speed and initial image 
        initialImg = caster.Execute( self.n2sInitialImg.sitkImg )
        speedImg   = caster.Execute( self.n2sSpeedlImg.sitkImg  )
        
        padder = sitk.ZeroFluxNeumannPadImageFilter()
        padder.SetPadLowerBound( [ 0, 0, self.paddingZ ] )
        padder.SetPadUpperBound( [ 0, 0, self.paddingZ ] )
        
        # Update the initial and the speed image with the padded version
        initialImg = padder.Execute( initialImg )
        speedImg   = padder.Execute( speedImg   )
        
        # Generate the level-set filter and feed it with the relevant parameters
        lsFilter = sitk.ShapeDetectionLevelSetImageFilter()
        lsFilter.SetCurvatureScaling( self.curvatureScaling )
        lsFilter.SetPropagationScaling( self.propagationScaling )
        lsFilter.SetNumberOfIterations( self.numberOfIterations )
        lsFilter.SetMaximumRMSError( self.maxRMSError )
        
        outLSImage = lsFilter.Execute( initialImg, speedImg )
        
        # Perform some post-processing steps (i.e. cropping)
        cropper = sitk.CropImageFilter()
        cropper.SetLowerBoundaryCropSize( [ 0, 0, self.paddingZ ] )
        cropper.SetUpperBoundaryCropSize( [ 0, 0, self.paddingZ ] )
        
        outLSImage = cropper.Execute( outLSImage )
        
        # convert the input image back to a nibabel one
        # Use the speed function since this is more likely to be of type double
        self.outLSImage = self.n2sSpeedlImg.pushSITKImageContentIntoOriginalNibabelData( outLSImage ) 
        
        
        
        
    def saveLSOutput(self, outFileName):
        
        if self.outLSImage is None:
            print('WARNING: output image does not exist')
            return 
        
        try:
            nib.save(self.outLSImage, outFileName )
        except:
            print('ERROR: could not write output image to file')
            
    
    
    
if __name__ == '__main__':
    
    # Some debugging parameters
    speedImgName   = 'C:/debugData/workflow-improv/preprocessOut/dvf_vec_frame1_to_frame2__labelOut.nii.gz'
    initialImgName = 'C:/debugData/workflow-improv/preprocessOut/dvf_vec_frame1_to_frame2__labelIn.nii.gz'
    lsOutImageName = 'C:/debugData/workflow-improv/preprocessOut/dvf_vec_frame1_to_frame2__LS.nii.gz'

    # Read the input images
    initialImg = sitk.ReadImage( initialImgName )
    speedImg   = sitk.ReadImage( speedImgName   )
        
    lse = LevelSetEvolution( initialImg, speedImg )
    lse.numberOfIterations = 100
    lse.run()
    lse.saveLSOutout( lsOutImageName )
    




