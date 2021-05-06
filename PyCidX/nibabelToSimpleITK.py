#!/usr/bin/env python3

import SimpleITK as sITK
import nibabel as nib
import numpy as np


class nibabelToSimpleITK(object):

    @staticmethod
    def sitkImageFromNib( nibImageIn ):
        '''
        Convert a nibabel image into an SITK object.

        @param nibImageIn: Image object read with nibabel library
        '''

        # Currently only 3d images are supported
        if ( nibImageIn.header['dim'][0] != 3):
            print("WARNING: This class is currently only intended for 3D images")

        # Generate an sitk image object from the nibabel image array,
        # Note that the order of the axes is reverted
        sitkImage = sITK.GetImageFromArray(np.transpose( nibImageIn.get_fdata(), [2, 1, 0]))

        # Set the image geometry
        # - origin
        sitkImage.SetOrigin(nibImageIn.affine[:3, 3] * np.array([-1, -1, 1]))

        # - spacing
        sitkImage.SetSpacing( nibImageIn.header['pixdim'][1:4].astype(np.double) )

        # - direction
        dirMatrix = nibImageIn.affine[:3, :3].copy()
        dirMatrix[:, 0] = dirMatrix[:, 0] / np.linalg.norm(dirMatrix[:, 0])
        dirMatrix[:, 1] = dirMatrix[:, 1] / np.linalg.norm(dirMatrix[:, 1])
        dirMatrix[:, 2] = dirMatrix[:, 2] / np.linalg.norm(dirMatrix[:, 2])

        dirMatrix[:2, :] = dirMatrix[:2, :] * (-1)

        sitkImage.SetDirection(dirMatrix.reshape(-1))

        return sitkImage


    @staticmethod
    def nibImageFromSITK( sITKImageIn ):
        '''
        Generate a new nifti image from a given SITK image.

        @param sITKImageIn: THe simple ITK image object to be converted. Note, only 3D images supported at the moment.
        '''

        # Currently only 3D images supported.
        if (sITKImageIn.GetDimension() != 3):
            print("WARNING: This class is currently only intended for 3D images")

        affineMatrix = np.eye(4)

        # Create the matrix according to itkSoftware guide
        affineMatrix[:3,:3] = np.dot( np.diag(sITKImageIn.GetSpacing()), np.array(sITKImageIn.GetDirection()).reshape([3,-1]) )
        affineMatrix[:3,3] = sITKImageIn.GetOrigin()

        # Account for change in geometry dicom/ITK vs. nifti
        affineMatrix[:2, :] = (-1) * affineMatrix[:2, :]

        return nib.Nifti1Image( np.transpose( sITK.GetArrayFromImage( sITKImageIn ), [2,1,0]), affineMatrix )



if __name__ == '__main__':
    
    # # Load the test image
    # nImg = nib.load( 'D:/debugData/test_01_MCR.nii.gz' )
    #
    # # Generate the converter and perform the conversion
    # converter = nibabelToSimpleITK( nImg )
    # convertedSITKImg = converter.convertToSITK()
    #
    # # Revert the changes and save the image
    # recoveredNibImg = converter.pushSITKImageContentIntoOriginalNibabelData(convertedSITKImg)
    # nib.save( recoveredNibImg, 'D:/debugData/test_01_MCR_recovered.nii.gz' )
    #

    # Conversion from SITK to nibabel
    nImg = nib.load( 'C:/debugData/cidX/resampled_only_follow_up_12.nii' )
    sReader = sITK.ImageFileReader()
    sReader.SetFileName('C:/debugData/cidX/resampled_only_follow_up_12.nii')
    sImg = sReader.Execute()

    converter = nibabelToSimpleITK( nImg )
    convertedSITKImg = converter.convertToSITK()

    sWriter = sITK.ImageFileWriter()
    sWriter.SetFileName( 'C:/debugData/cidX/resampled_only_follow_up_12_recSITK.nii.gz' )
    sWriter.Execute( convertedSITKImg )
    recoveredNibImg = nibabelToSimpleITK.nibImageFromSITK( convertedSITKImg )
    nib.save(recoveredNibImg, 'C:/debugData/cidX/resampled_only_follow_up_12_recNib.nii.gz')

    cropper = sITK.CropImageFilter()
    cropper.SetUpperBoundaryCropSize([1, 2, 3])
    cropper.SetLowerBoundaryCropSize([4, 5, 6])
    cImg = cropper.Execute(convertedSITKImg)
    ncImg = nibabelToSimpleITK.nibImageFromSITK(cImg)
    nib.save(ncImg, 'C:/debugData/cidX/resampled_only_follow_up_12_recNib_cropped.nii.gz')

