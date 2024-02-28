#!/usr/bin/env python3

# My imports
import PyCidX.nibabelToSimpleITK as n2sConv
import SimpleITK as sitk
import nibabel as nib
import numpy as np


class LevelSetEvolution(object):

    def __init__( self, initialNibabelImage, speedNibabelImage ):
        """
        Generate the level-set evolution object. It takes two images of the same size defining the speed of the levelset
        as well as the initial

        :param initialNibabelImage: The nibabel image holding the initial levelset, i.e. contour defined by the zero-
                                    crossing
        :param speedNibabelImage: The nibabel image holding the speed function.
        """

        # The images
        # Only keep the converter objects
        self.initialNibIamge = initialNibabelImage
        self.speedNibImage = speedNibabelImage

        # Default parameters
        self.maxRMSError        = 0.0005
        self.propagationScaling = 1.0
        self.curvatureScaling   = 20.0
        self.numberOfIterations = 5000
        
        self.paddingZ = 5
        
        # The output - set to non such that it is not written without being calculated...
        self.outLSImage = None

    
    
    
    def run(self):
        """
        Run the relevant pre-processing of the given images and eventually the level-set evolution.
        """

        # Pre-process: Convert images to
        #    1) simpleITK images
        #    2) float32 and
        #    3) pad them with the zero-flux padding filter in the lower and upper z-direction
        #    4) resample to isotropic (lowest) resolution
        # self.n2sInitialImg.convertToSITK()
        # self.n2sSpeedlImg.convertToSITK()

        # Generate the caster object
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkFloat32)
        
        # local SimpleITK version of speed and initial image 
        initialImg = caster.Execute( n2sConv.nibabelToSimpleITK.sitkImageFromNib( self.initialNibIamge ) )
        speedImg   = caster.Execute( n2sConv.nibabelToSimpleITK.sitkImageFromNib( self.speedNibImage )  )

        # Generate the image padder
        padder = sitk.ZeroFluxNeumannPadImageFilter()
        padder.SetPadLowerBound( [ 0, 0, self.paddingZ ] )
        padder.SetPadUpperBound( [ 0, 0, self.paddingZ ] )
        
        # Update the initial and the speed image with the padded version
        initialPadImg = padder.Execute( initialImg )
        speedImg      = padder.Execute( speedImg   )

        # Resample to isotropic image resolution
        # resample to the lowest resolution
        maxSpacing = np.max( initialImg.GetSpacing() )
        initialPadImg = self._resampleToIsotropicResolution( initialPadImg, maxSpacing )
        speedImg      = self._resampleToIsotropicResolution( speedImg, maxSpacing   )

        # Generate the level-set filter and feed it with the relevant parameters
        lsFilter = sitk.ShapeDetectionLevelSetImageFilter()
        lsFilter.SetCurvatureScaling( self.curvatureScaling )
        lsFilter.SetPropagationScaling( self.propagationScaling )
        lsFilter.SetNumberOfIterations( self.numberOfIterations )
        lsFilter.SetMaximumRMSError( self.maxRMSError )
        # eventually run the level-set evolution
        outLSImage = lsFilter.Execute( initialPadImg, speedImg )
        
        # Perform some post-processing steps (i.e. resample back to the original image geometry)
        # Resample(Image image1, Image referenceImage, Transform transform, itk::simple::InterpolatorEnum interpolator, double defaultPixelValue=0.0, itk::simple::PixelIDValueEnum outputPixelType) -> Image
        trafo = sitk.Euler3DTransform()
        trafo.SetIdentity()
        outLSImage = sitk.Resample( outLSImage, initialImg, trafo, sitk.sitkLinear )

        # convert the input image back to a nibabel one
        self.outLSImage = n2sConv.nibabelToSimpleITK.nibImageFromSITK( outLSImage )
        # Use the speed function since this is more likely to be of type double
        #self.outLSImage = self.n2sSpeedlImg.pushSITKImageContentIntoOriginalNibabelData( outLSImage )
        
        
    def _resampleToIsotropicResolution(self, sitkImageIn, resolutionInMM):
        """
        Resample the given input image to the given resolution. Keep the origin, spacing and direction as in the given
        image. For now 3d input images and nearest-neighbour interpolation will be assumed.

        :param sitkImageIn: The image that will be resampled.
        :param resolutionInMM: The scalar of the final resolution of the returned image
        :return: Resampled SITK image object
        """

        # Get the image size and spacing of the input image
        origSize = sitkImageIn.GetSize()
        origSpacing = sitkImageIn.GetSpacing()

        # Define the output spacing and size
        outSpacing = [resolutionInMM, resolutionInMM, resolutionInMM]
        outSize = [int(np.ceil(origSize[0] * origSpacing[0] / outSpacing[0])),
                   int(np.ceil(origSize[1] * origSpacing[1] / outSpacing[1])),
                   int(np.ceil(origSize[2] * origSpacing[2] / outSpacing[2]))]

        # Generate a simple transform that will be set to identity
        idTransform = sitk.Euler3DTransform()
        idTransform.SetIdentity()

        # Resample
        resampledImg = sitk.Resample( sitkImageIn, outSize, idTransform,
                                      sitk.sitkNearestNeighbor,
                                      sitkImageIn.GetOrigin(), outSpacing, sitkImageIn.GetDirection() )

        return resampledImg



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
    initialImg = nib.load( initialImgName )
    speedImg = nib.load( speedImgName )

    lse = LevelSetEvolution( initialImg, speedImg )
    lse.numberOfIterations = 100
    lse.run()
    nib.save( lse.outLSImage, lsOutImageName )
