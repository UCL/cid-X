#!/usr/bin/env python3

import sys
import os
import nibabel as nib
from scipy.ndimage import distance_transform_edt, gaussian_filter
import numpy as np
import PyCidX.displacementVectorFieldTools as dvfTools
import PyCidX.commandExecution as commandExecution  
import PyCidX.finiteDifferences as fd
import PyCidX.diffuseFixedValuesIntoMask as diffusor
import PyCidX.findExecutable as findExecutable
import PyCidX.signedDistanceMap as sdt


def _processDVF( mask, voxelSize, dvfExt, fallOffDist, gaussianPreSmoothingRadius = 5.0):
    
    # Extend the DVF 
    distances, indices = distance_transform_edt( 1-mask, voxelSize, return_distances=True, return_indices=True )

    dvfExt[:,:,:,0,0] = dvfExt[:,:,:,0,0][ indices[0,:,:,:],indices[1,:,:,:],indices[2,:,:,:] ]
    dvfExt[:,:,:,0,1] = dvfExt[:,:,:,0,1][ indices[0,:,:,:],indices[1,:,:,:],indices[2,:,:,:] ]
    dvfExt[:,:,:,0,2] = dvfExt[:,:,:,0,2][ indices[0,:,:,:],indices[1,:,:,:],indices[2,:,:,:] ]
    del indices
    
    # Let the displacements fall off to zero between fallOffDist and 2*fallOffDist
    dvfExt[:,:,:,0,0][np.where(distances > 2*fallOffDist)] = 0.0
    dvfExt[:,:,:,0,1][np.where(distances > 2*fallOffDist)] = 0.0
    dvfExt[:,:,:,0,2][np.where(distances > 2*fallOffDist)] = 0.0
    dvfExt[:,:,:,0,0][np.where(distances>fallOffDist)] = dvfExt[:,:,:,0,0][np.where(distances>fallOffDist)] * (2-(distances[np.where(distances>fallOffDist)]/fallOffDist))
    dvfExt[:,:,:,0,1][np.where(distances>fallOffDist)] = dvfExt[:,:,:,0,1][np.where(distances>fallOffDist)] * (2-(distances[np.where(distances>fallOffDist)]/fallOffDist))
    dvfExt[:,:,:,0,2][np.where(distances>fallOffDist)] = dvfExt[:,:,:,0,2][np.where(distances>fallOffDist)] * (2-(distances[np.where(distances>fallOffDist)]/fallOffDist))
    del distances
    
    # Smooth the extended DVF
    dvfExt[:,:,:,0,0] = gaussian_filter( dvfExt[:,:,:,0,0], gaussianPreSmoothingRadius )
    dvfExt[:,:,:,0,1] = gaussian_filter( dvfExt[:,:,:,0,1], gaussianPreSmoothingRadius )
    dvfExt[:,:,:,0,2] = gaussian_filter( dvfExt[:,:,:,0,2], gaussianPreSmoothingRadius )
            
    # Check that no folding exists in the extended DVF
    dvfTool=dvfTools.DisplacementVectorFieldTools3D( voxelSize[0], voxelSize[1], voxelSize[2])
    dvfTool.minAllowedJacobian = 0.025
    dvfTool.smoothingRadius = 2.0
    dvfTool.verbose = True
    dvfExt[:,:,:,0,0], dvfExt[:,:,:,0,1], dvfExt[:,:,:,0,2] = dvfTool.correctFolding( dvfExt[:,:,:,0,0]*(-1) , 
                                                                                      dvfExt[:,:,:,0,1]*(-1), 
                                                                                      dvfExt[:,:,:,0,2], False )
    
    dvfExt[:,:,:,0,0] = dvfExt[:,:,:,0,0]*(-1)
    dvfExt[:,:,:,0,1] = dvfExt[:,:,:,0,1]*(-1)
    
    return dvfExt




def XCATdvfPostProcessing( XCATDVFNiiFileName, 
                           sdt_1_fileName, 
                           sdtDx_1_fileName, sdtDy_1_fileName, sdtDz_1_fileName, 
                           outDir, tmpDir=None, numProcessorsToUse=None, 
                           niftyRegBinDirIn='' ):
    #
    # Post-process the XCAT DVF files
    #
    if tmpDir is None:
        tmpDir = outDir
    
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)
    
    print("Post-processing XCAT DVF image ")
    print(" input DVF:                     " + XCATDVFNiiFileName   )
    print(" input inside/outside SDT:      " + sdt_1_fileName )
    print(" input inside/outside SDT (dx): " + sdtDx_1_fileName )
    print(" input inside/outside SDT (dy): " + sdtDy_1_fileName )
    print(" input inside/outside SDT (dz): " + sdtDz_1_fileName )
    print(" output directory:              " + outDir               )
    print(" temp directory:                " + tmpDir               )

    fallOffDist = 5.0
    guassianPreSmoothingRadius = 4.0
    generateCorrected1toN = True

    print( " fall-off distance: %f" %(fallOffDist) )
    print( " gaussian pre-smoothing radius: %f" %(guassianPreSmoothingRadius) )
    if generateCorrected1toN:
        print( " generate corrected 1-to-N DVF: true"  )
    else:
        print( " generate corrected 1-to-N DVF: false"  )
        
    # Set the number of processors
    niftyReg_transformAdditionalParams = ''
    if numProcessorsToUse is not None:
        niftyReg_transformAdditionalParams += ' -omp %i ' % numProcessorsToUse
    
    # Extract the base file name so that all files can be named according to the input file
    DVFBaseFileName = os.path.split(XCATDVFNiiFileName)[1].split('.nii')[0]
    
    # output image file names needs to be cleaned up afterwards
    dvfInExtended_1toN_fileName   = tmpDir + DVFBaseFileName + '_dvfIn.nii.gz'
    dvfOutExtended_1toN_fileName  = tmpDir + DVFBaseFileName + '_dvfOut.nii.gz'
    dvfInExt_Nto1_fileName        = tmpDir + DVFBaseFileName + '_dvfIn_inv.nii.gz'
    dvfOutExt_Nto1_fileName       = tmpDir + DVFBaseFileName + '_dvfOut_inv.nii.gz'
    sdtIn_N_fileName              = tmpDir + DVFBaseFileName + '_sdt_warpedWith_dvfIn_inv.nii.gz'
    sdtOut_N_fileName             = tmpDir + DVFBaseFileName + '_sdt_warpedWith_dvfOut_inv.nii.gz'
    sdtInOutProd_N_fileName       = tmpDir + DVFBaseFileName + '_sdt_warpedWith_dvfs_inv_mult.nii.gz'
    sdtInOutSum_N_fileName        = tmpDir + DVFBaseFileName + '_sdt_warpedWith_dvfs_inv_sum.nii.gz'
    sdt_N_NiiFileName             = tmpDir + DVFBaseFileName + '_sdt_warpedWith_dvfs_inv_sum_thresh_sdt_smooth.nii.gz'

    sdt1dx_in_N_fileName          = tmpDir + DVFBaseFileName + '_sdtdx1_in_N.nii.gz'
    sdt1dy_in_N_fileName          = tmpDir + DVFBaseFileName + '_sdtdy1_in_N.nii.gz'
    sdt1dz_in_N_fileName          = tmpDir + DVFBaseFileName + '_sdtdz1_in_N.nii.gz'
    
    sdt1dx_out_N_fileName         = tmpDir + DVFBaseFileName + '_sdtdx1_out_N.nii.gz'
    sdt1dy_out_N_fileName         = tmpDir + DVFBaseFileName + '_sdtdy1_out_N.nii.gz'
    sdt1dz_out_N_fileName         = tmpDir + DVFBaseFileName + '_sdtdz1_out_N.nii.gz'

    dvfCor_Nto1_fileName          = outDir + DVFBaseFileName + '_dvfCor_Nto1.nii.gz'    # Main output

    dvfInExtCor_Nto1_fileName     = tmpDir + DVFBaseFileName + '_dvfCorIn_Nto1.nii.gz'
    dvfOutExtCor_Nto1_fileName    = tmpDir + DVFBaseFileName + '_dvfCorOut_Nto1.nii.gz'
    
    dvfInExtCor_1toN_fileName     = tmpDir + DVFBaseFileName + '_dvfCorIn_1toN.nii.gz'
    dvfOutExtCor_1toN_fileName    = tmpDir + DVFBaseFileName + '_dvfCorOut_1toN.nii.gz'
    dvfCor_1toN_fileName          = outDir + DVFBaseFileName + '_dvfCor_1toN.nii.gz'    # Main output, inverse

    tmpFileList = []
    tmpFileList.append( dvfInExtended_1toN_fileName )
    tmpFileList.append( dvfOutExtended_1toN_fileName )
    tmpFileList.append( dvfInExt_Nto1_fileName )
    tmpFileList.append( dvfOutExt_Nto1_fileName )
    tmpFileList.append( sdtIn_N_fileName )
    tmpFileList.append( sdtOut_N_fileName )
    tmpFileList.append( sdtInOutProd_N_fileName )
    tmpFileList.append( sdtInOutSum_N_fileName ) 
    tmpFileList.append( sdt_N_NiiFileName )
    tmpFileList.append( sdt1dx_in_N_fileName )
    tmpFileList.append( sdt1dy_in_N_fileName )
    tmpFileList.append( sdt1dz_in_N_fileName )
    tmpFileList.append( sdt1dx_out_N_fileName )
    tmpFileList.append( sdt1dy_out_N_fileName )
    tmpFileList.append( sdt1dz_out_N_fileName )
    tmpFileList.append( dvfInExtCor_Nto1_fileName )
    tmpFileList.append( dvfOutExtCor_Nto1_fileName )
    tmpFileList.append( dvfInExtCor_1toN_fileName )
    tmpFileList.append( dvfOutExtCor_1toN_fileName )
    
    # Prepare the processing
    if not os.path.exists( outDir ):
        print("  ... -> generated output directory")
        os.makedirs( outDir )
    
    # Load the XCAT DVF and signed distance function that defines the inside/outside region
    dvf_1toN_Nii = nib.load( XCATDVFNiiFileName )
    sdt_1_Nii    = nib.load( sdt_1_fileName )
    try:
        voxelSize    = sdt_1_Nii.header['pixdim'][1:4]
    except:
        voxelSize    = sdt_1_Nii.get_header()['pixdim'][1:4]
    
    # Check if the spatial derivatives of the signed distance function exist
    # if not, calculate them here and save them for later use
    if not os.path.exists( sdtDx_1_fileName ):
        print("Note: Attempting to generate Dx of sdt at time-point 1")

        # Calculate the gradient of the original SDT
        finDif = fd.finiteDifferences3D( voxelSize[0], voxelSize[1], voxelSize[2], 'central' )
        sdt_1_dx = np.zeros_like( sdt_1_Nii.get_fdata() )
        finDif.diffX( sdt_1_Nii.get_fdata(), sdt_1_dx )
        try: 
            sdtDx_1_nii = nib.Nifti1Image( sdt_1_dx, sdt_1_Nii.affine, sdt_1_Nii.header )
        except:
            sdtDx_1_nii = nib.Nifti1Image( sdt_1_dx, sdt_1_Nii.get_affine(), sdt_1_Nii.get_header() )
        nib.save( sdtDx_1_nii, sdtDx_1_fileName )
        del finDif, sdtDx_1_nii, sdt_1_dx
        
    if not os.path.exists( sdtDy_1_fileName ):
        print("Note: Attempting to generate Dy of sdt at time-point 1")

        # Calculate the gradient of the original SDT
        finDif = fd.finiteDifferences3D( voxelSize[0], voxelSize[1], voxelSize[2], 'central' )
        sdt_1_dy = np.zeros_like( sdt_1_Nii.get_fdata() )
        finDif.diffY( sdt_1_Nii.get_fdata(), sdt_1_dy )
        try:
            sdtDy_1_nii = nib.Nifti1Image( sdt_1_dy, sdt_1_Nii.affine, sdt_1_Nii.header )
        except:
            sdtDy_1_nii = nib.Nifti1Image( sdt_1_dy, sdt_1_Nii.get_affine(), sdt_1_Nii.get_header() )
        nib.save( sdtDy_1_nii, sdtDy_1_fileName )
        del finDif, sdtDy_1_nii, sdt_1_dy

    if not os.path.exists( sdtDz_1_fileName ):
        print("Note: Attempting to generate Dz of sdt at time-point 1")

        # Calculate the gradient of the original SDT
        finDif = fd.finiteDifferences3D( voxelSize[0], voxelSize[1], voxelSize[2], 'central' )
        sdt_1_dz = np.zeros_like( sdt_1_Nii.get_fdata() )
        finDif.diffZ( sdt_1_Nii.get_fdata(), sdt_1_dz )
        try:
            sdtDz_1_nii = nib.Nifti1Image( sdt_1_dz, sdt_1_Nii.affine, sdt_1_Nii.header )
        except:
            sdtDz_1_nii = nib.Nifti1Image( sdt_1_dz, sdt_1_Nii.get_affine(), sdt_1_Nii.get_header() )
        nib.save( sdtDz_1_nii, sdtDz_1_fileName )
        del finDif, sdtDz_1_nii, sdt_1_dz
    
    
    # Split the DVF into inside/outside and extend, then save as nifti images
    print("  ... -> extending inside DVF")
    dvfInExtended_1toN  = _processDVF( sdt_1_Nii.get_fdata() <  0, voxelSize, dvf_1toN_Nii.get_fdata().copy(), fallOffDist, guassianPreSmoothingRadius )
    print("  ... -> extending outside DVF")
    dvfOutExtended_1toN = _processDVF( sdt_1_Nii.get_fdata() >= 0, voxelSize, dvf_1toN_Nii.get_fdata().copy(), fallOffDist, guassianPreSmoothingRadius )
    try:
        dvfInExtended_1toN_Nii  = nib.Nifti1Image( dvfInExtended_1toN,  dvf_1toN_Nii.affine, header=dvf_1toN_Nii.header )
        dvfOutExtended_1toN_Nii = nib.Nifti1Image( dvfOutExtended_1toN, dvf_1toN_Nii.affine, header=dvf_1toN_Nii.header )
    except:
        dvfInExtended_1toN_Nii  = nib.Nifti1Image( dvfInExtended_1toN,  dvf_1toN_Nii.get_affine(), header=dvf_1toN_Nii.get_header() )
        dvfOutExtended_1toN_Nii = nib.Nifti1Image( dvfOutExtended_1toN, dvf_1toN_Nii.get_affine(), header=dvf_1toN_Nii.get_header() )
    nib.save( dvfInExtended_1toN_Nii,  dvfInExtended_1toN_fileName  )
    nib.save( dvfOutExtended_1toN_Nii, dvfOutExtended_1toN_fileName )
    
    del dvfInExtended_1toN_Nii
    del dvfOutExtended_1toN_Nii
    del dvfInExtended_1toN
    del dvfOutExtended_1toN 
    
    # Invert both extended (push #1 -> #N) DVFs
    print("  ... -> calculating inverse of inside transform")
    niftyReg_transformCMD = niftyRegBinDirIn + 'reg_transform'
    niftyReg_transformParams  = '-invNrr' 
    niftyReg_transformParams +=  ' ' + dvfInExtended_1toN_fileName 
    niftyReg_transformParams +=  ' ' + sdt_1_fileName 
    niftyReg_transformParams +=  ' ' + dvfInExt_Nto1_fileName 
    niftyReg_transformParams +=  niftyReg_transformAdditionalParams 
    commandExecution.runCommand( niftyReg_transformCMD, niftyReg_transformParams, workDir=outDir )

    print("  ... -> calculating inverse of outside transform")
    niftyReg_transformParams  = '-invNrr' 
    niftyReg_transformParams +=  ' ' + dvfOutExtended_1toN_fileName 
    niftyReg_transformParams +=  ' ' + sdt_1_fileName 
    niftyReg_transformParams +=  ' ' + dvfOutExt_Nto1_fileName 
    niftyReg_transformParams +=  niftyReg_transformAdditionalParams 
    commandExecution.runCommand(niftyReg_transformCMD, niftyReg_transformParams, workDir=outDir )
    
    # Transform the mask image with the inverse DVFs
    print("  ... -> resampling SDT to time point N with inverse inside transform")
    niftyReg_resampleCMD = niftyRegBinDirIn + 'reg_resample'
    niftyReg_resampleParams  = ' -ref '   + sdt_1_fileName
    niftyReg_resampleParams += ' -flo '   + sdt_1_fileName
    niftyReg_resampleParams += ' -trans ' + dvfInExt_Nto1_fileName
    niftyReg_resampleParams += ' -res '   + sdtIn_N_fileName
    commandExecution.runCommand(niftyReg_resampleCMD, niftyReg_resampleParams, workDir=outDir )
 
    print("  ... -> resampling SDT to time point N with inverse outside transform")
    niftyReg_resampleParams  = ' -ref '   + sdt_1_fileName
    niftyReg_resampleParams += ' -flo '   + sdt_1_fileName
    niftyReg_resampleParams += ' -res '   + sdtOut_N_fileName
    niftyReg_resampleParams += ' -trans ' + dvfOutExt_Nto1_fileName
    commandExecution.runCommand(niftyReg_resampleCMD, niftyReg_resampleParams, workDir=outDir)
    
    # Multiply the transformed SDTs to determine gap and overlap regions
    print("  ... -> calculating gaps and overlaps")
    sdtIn_N_nii  = nib.load( sdtIn_N_fileName  )
    sdtOut_N_nii = nib.load( sdtOut_N_fileName )
    
    # Calculate the gap and overlap measures and save as nifti images 
    # pulled into #N
    sdtInOutProd_N = sdtIn_N_nii.get_fdata() * sdtOut_N_nii.get_fdata()
    sdtInOutSum_N  = sdtIn_N_nii.get_fdata() + sdtOut_N_nii.get_fdata()
    
    try:
        sdtInOutProd_N_nii = nib.Nifti1Image( sdtInOutProd_N, sdtIn_N_nii.affine )
        sdtInOutSum_N_nii  = nib.Nifti1Image( sdtInOutSum_N, sdtIn_N_nii.affine  )
    except:
        sdtInOutProd_N_nii = nib.Nifti1Image( sdtInOutProd_N, sdtIn_N_nii.get_affine() )
        sdtInOutSum_N_nii  = nib.Nifti1Image( sdtInOutSum_N, sdtIn_N_nii.get_affine()  )
#     nib.save( sdtInOutProd_N_nii, sdtInOutProd_N_fileName )
#     nib.save( sdtInOutSum_N_nii,  sdtInOutSum_N_fileName )
    
    # Generate a signed distance function 
    sdtMapper = sdt.SignedDistanceMap( sdtInOutSum_N_nii )
    sdtMapper.run()
    nib.save( sdtMapper.distMapImg, sdt_N_NiiFileName )
    
    
    print("  ... -> correcting gaps and overlaps")
    print("  ... -> resampling gradient to N (inside)")
    # This gradient needs to be transformed to time point N for correction purposes
    niftyReg_resampleParams  = ' -ref '   + sdtDx_1_fileName
    niftyReg_resampleParams += ' -flo '   + sdtDx_1_fileName
    niftyReg_resampleParams += ' -res '   + sdt1dx_in_N_fileName
    niftyReg_resampleParams += ' -trans ' + dvfInExt_Nto1_fileName
    commandExecution.runCommand( niftyReg_resampleCMD, niftyReg_resampleParams, workDir=outDir )
    
    niftyReg_resampleParams  = ' -ref '   + sdtDy_1_fileName
    niftyReg_resampleParams += ' -flo '   + sdtDy_1_fileName
    niftyReg_resampleParams += ' -res '   + sdt1dy_in_N_fileName
    niftyReg_resampleParams += ' -trans ' + dvfInExt_Nto1_fileName
    commandExecution.runCommand( niftyReg_resampleCMD, niftyReg_resampleParams, workDir=outDir )
    
    niftyReg_resampleParams  = ' -ref '   + sdtDz_1_fileName
    niftyReg_resampleParams += ' -flo '   + sdtDz_1_fileName
    niftyReg_resampleParams += ' -res '   + sdt1dz_in_N_fileName
    niftyReg_resampleParams += ' -trans ' + dvfInExt_Nto1_fileName
    commandExecution.runCommand( niftyReg_resampleCMD, niftyReg_resampleParams, workDir=outDir )
    
    
    print("  ... -> resampling gradient to N (outside)")
    # This gradient needs to be transformed to time point N for correction purposes
    niftyReg_resampleParams  = ' -ref '   + sdtDx_1_fileName
    niftyReg_resampleParams += ' -flo '   + sdtDx_1_fileName
    niftyReg_resampleParams += ' -res '   + sdt1dx_out_N_fileName
    niftyReg_resampleParams += ' -trans ' + dvfOutExt_Nto1_fileName
    commandExecution.runCommand( niftyReg_resampleCMD, niftyReg_resampleParams, workDir=outDir )
    
    niftyReg_resampleParams  = ' -ref '   + sdtDy_1_fileName
    niftyReg_resampleParams += ' -flo '   + sdtDy_1_fileName
    niftyReg_resampleParams += ' -res '   + sdt1dy_out_N_fileName
    niftyReg_resampleParams += ' -trans ' + dvfOutExt_Nto1_fileName
    commandExecution.runCommand( niftyReg_resampleCMD, niftyReg_resampleParams, workDir=outDir )
    
    niftyReg_resampleParams  = ' -ref '   + sdtDz_1_fileName
    niftyReg_resampleParams += ' -flo '   + sdtDz_1_fileName
    niftyReg_resampleParams += ' -res '   + sdt1dz_out_N_fileName
    niftyReg_resampleParams += ' -trans ' + dvfOutExt_Nto1_fileName
    commandExecution.runCommand( niftyReg_resampleCMD, niftyReg_resampleParams, workDir=outDir )
    
    # Correct the individual inverted DVFs for gaps and overlaps
    sdt1dx_in_N_nii  = nib.load( sdt1dx_in_N_fileName )
    sdt1dy_in_N_nii  = nib.load( sdt1dy_in_N_fileName )
    sdt1dz_in_N_nii  = nib.load( sdt1dz_in_N_fileName )
    sdt1dx_out_N_nii = nib.load( sdt1dx_out_N_fileName )
    sdt1dy_out_N_nii = nib.load( sdt1dy_out_N_fileName )
    sdt1dz_out_N_nii = nib.load( sdt1dz_out_N_fileName )
    
    # Calculate a weight for the sliding correction using a diffusion process 
    # with enforced values where gaps/overlaps occur
    sdt_N_nii = nib.load( sdt_N_NiiFileName )
    
    distDiffIn_N  = sdtIn_N_nii.get_fdata()  - sdt_N_nii.get_fdata()
    distDiffOut_N = sdtOut_N_nii.get_fdata() - sdt_N_nii.get_fdata()
    
    del sdtIn_N_nii, sdtOut_N_nii
    
    print("  ... -> diffusing correction factor for sliding-preserving correction")
    newDistEnforcementFactor = np.zeros_like( sdt_N_nii.get_fdata() )
    idxToCorrect = np.where(sdtInOutProd_N_nii.get_fdata() < 0)
    newDistEnforcementFactor[idxToCorrect] = 1
    
    dfvim = diffusor.diffuseFixedValuesIntoMask(newDistEnforcementFactor, newDistEnforcementFactor, voxelSize, 50)
    newDistEnforcementFactor = dfvim.diffuse()

    # Correct for sliding in DVF_Nto1
    corFactIn  = newDistEnforcementFactor * distDiffIn_N
    corFactOut = newDistEnforcementFactor * distDiffOut_N
    
    del newDistEnforcementFactor, distDiffIn_N, distDiffOut_N
     
    normFac = np.sqrt(sdt1dx_in_N_nii.get_fdata()**2 + sdt1dy_in_N_nii.get_fdata()**2+sdt1dz_in_N_nii.get_fdata()**2)
    normFac[np.where(normFac==0)]=1
    
    dvfInExt_Nto1_nii  = nib.load( dvfInExt_Nto1_fileName  )
    dvfInExtCor_Nto1 = dvfInExt_Nto1_nii.get_fdata().copy()
    dvfInExtCor_Nto1[:,:,:,0,0] = dvfInExtCor_Nto1[:,:,:,0,0] + corFactIn * sdt1dx_in_N_nii.get_fdata() / normFac
    dvfInExtCor_Nto1[:,:,:,0,1] = dvfInExtCor_Nto1[:,:,:,0,1] + corFactIn * sdt1dy_in_N_nii.get_fdata() / normFac
    dvfInExtCor_Nto1[:,:,:,0,2] = dvfInExtCor_Nto1[:,:,:,0,2] - corFactIn * sdt1dz_in_N_nii.get_fdata() / normFac
    
    del sdt1dx_in_N_nii, sdt1dy_in_N_nii, sdt1dz_in_N_nii, normFac, corFactIn
    
    normFac = np.sqrt(sdt1dx_out_N_nii.get_fdata()**2 + sdt1dy_out_N_nii.get_fdata()**2+sdt1dz_out_N_nii.get_fdata()**2)
    normFac[np.where(normFac==0)]=1
    
    dvfOutExt_Nto1_nii = nib.load( dvfOutExt_Nto1_fileName )
    dvfOutExtCor_Nto1 = dvfOutExt_Nto1_nii.get_fdata().copy()
    dvfOutExtCor_Nto1[:,:,:,0,0] = dvfOutExtCor_Nto1[:,:,:,0,0] + corFactOut * sdt1dx_out_N_nii.get_fdata() / normFac
    dvfOutExtCor_Nto1[:,:,:,0,1] = dvfOutExtCor_Nto1[:,:,:,0,1] + corFactOut * sdt1dy_out_N_nii.get_fdata() / normFac
    dvfOutExtCor_Nto1[:,:,:,0,2] = dvfOutExtCor_Nto1[:,:,:,0,2] - corFactOut * sdt1dz_out_N_nii.get_fdata() / normFac
    
    del sdt1dx_out_N_nii, sdt1dy_out_N_nii, sdt1dz_out_N_nii, normFac, corFactOut
    
    # Copy inside DVF, fill in the out DVF where appropriate
    dvfCor_Nto1 = dvfInExtCor_Nto1.copy()
    idxOutside=np.where( sdt_N_nii.get_fdata()>=0 )
    dvfCor_Nto1[:,:,:,0,0][idxOutside] = dvfOutExtCor_Nto1[:,:,:,0,0][idxOutside] 
    dvfCor_Nto1[:,:,:,0,1][idxOutside] = dvfOutExtCor_Nto1[:,:,:,0,1][idxOutside] 
    dvfCor_Nto1[:,:,:,0,2][idxOutside] = dvfOutExtCor_Nto1[:,:,:,0,2][idxOutside] 
    
    del sdt_N_nii
    
    try:    
        dvfCor_Nto1_nii = nib.Nifti1Image( dvfCor_Nto1, affine=dvfInExt_Nto1_nii.affine, header=dvfInExt_Nto1_nii.header  )
    except:
        dvfCor_Nto1_nii = nib.Nifti1Image( dvfCor_Nto1, affine=dvfInExt_Nto1_nii.get_affine(), header=dvfInExt_Nto1_nii.get_header()  )
    nib.save(dvfCor_Nto1_nii, dvfCor_Nto1_fileName)
    
    del dvfCor_Nto1, dvfCor_Nto1_nii
    
    if generateCorrected1toN:
        # Finally generate the inverse of the corrected N-to-1
        # Save the corrected inside/outside DVFs
        print("  ... -> updating the transformation 1->N using the previous corrections")
        try:
            dvfInExtCor_Nto1_nii  = nib.Nifti1Image( dvfInExtCor_Nto1,  dvfInExt_Nto1_nii.affine,  header=dvfInExt_Nto1_nii.header  )
            dvfOutExtCor_Nto1_nii = nib.Nifti1Image( dvfOutExtCor_Nto1, dvfOutExt_Nto1_nii.affine, header=dvfOutExt_Nto1_nii.header )
        except:
            dvfInExtCor_Nto1_nii  = nib.Nifti1Image( dvfInExtCor_Nto1,  dvfInExt_Nto1_nii.get_affine(),  header=dvfInExt_Nto1_nii.get_header()  )
            dvfOutExtCor_Nto1_nii = nib.Nifti1Image( dvfOutExtCor_Nto1, dvfOutExt_Nto1_nii.get_affine(), header=dvfOutExt_Nto1_nii.get_header() )
        
        nib.save( dvfInExtCor_Nto1_nii,  dvfInExtCor_Nto1_fileName  )
        nib.save( dvfOutExtCor_Nto1_nii, dvfOutExtCor_Nto1_fileName )
        
        del dvfInExtCor_Nto1, dvfInExtCor_Nto1_nii, dvfOutExtCor_Nto1, dvfOutExtCor_Nto1_nii
        
        # Invert both extended (push #1 -> #N) DVFs
        niftyReg_transformParams  = '-invNrr' 
        niftyReg_transformParams +=  ' ' + dvfInExtCor_Nto1_fileName 
        niftyReg_transformParams +=  ' ' + sdt_1_fileName 
        niftyReg_transformParams +=  ' ' + dvfInExtCor_1toN_fileName
        niftyReg_transformParams +=  niftyReg_transformAdditionalParams 
        commandExecution.runCommand( niftyReg_transformCMD, niftyReg_transformParams, workDir=outDir )

        niftyReg_transformParams  = '-invNrr' 
        niftyReg_transformParams +=  ' ' + dvfOutExtCor_Nto1_fileName 
        niftyReg_transformParams +=  ' ' + sdt_1_fileName 
        niftyReg_transformParams +=  ' ' + dvfOutExtCor_1toN_fileName
        niftyReg_transformParams +=  niftyReg_transformAdditionalParams 
        commandExecution.runCommand( niftyReg_transformCMD, niftyReg_transformParams, workDir=outDir )
        
        dvfInExtCor_1toN_nii  = nib.load( dvfInExtCor_1toN_fileName  )
        dvfOutExtCor_1toN_nii = nib.load( dvfOutExtCor_1toN_fileName )
        idxOutside = np.where( sdt_1_Nii.get_fdata() >= 0 )
        
        # copy inside, replace outside
        dvfCor_1toN = dvfInExtCor_1toN_nii.get_fdata().copy()
        dvfCor_1toN[:,:,:,0,0][idxOutside]  = dvfOutExtCor_1toN_nii.get_fdata()[:,:,:,0,0][idxOutside]
        dvfCor_1toN[:,:,:,0,1][idxOutside]  = dvfOutExtCor_1toN_nii.get_fdata()[:,:,:,0,1][idxOutside]
        dvfCor_1toN[:,:,:,0,2][idxOutside]  = dvfOutExtCor_1toN_nii.get_fdata()[:,:,:,0,2][idxOutside]
        
        try:
            dvfCor_1toN_nii = nib.Nifti1Image( dvfCor_1toN, dvfInExt_Nto1_nii.affine, header=dvfInExt_Nto1_nii.header )
        except:
            dvfCor_1toN_nii = nib.Nifti1Image( dvfCor_1toN, dvfInExt_Nto1_nii.get_affine(), header=dvfInExt_Nto1_nii.get_header() )
        nib.save( dvfCor_1toN_nii, dvfCor_1toN_fileName )

    print("Removing temporary files:")
    for fileToDelete in tmpFileList:
        if os.path.exists(fileToDelete):
            print(" -> " + fileToDelete )
            os.remove(fileToDelete)
    


if __name__ == "__main__":
    
    print("Tool to post-process an XCAT DVF nifti volume.")
    print("  Output: ...")
    
    if len(sys.argv) < 3:
        print("Usage: XCATdvfPostProcessing.py pathToXCATDVFNiftiImage pathToXCATDVFInOutMask")
        print("        - pathToXCATDVFNiftiImage  -> Path to the DVF text file output by XCAT")
        print("        - pathToXCATDVFInOutSDT    -> Path to the file with the signed distance function determining in/out region.")
        print("        - pathToXCATDVFInOutSDT-dx -> Path to the file with the signed distance function determining in/out region.")
        print("        - pathToXCATDVFInOutSDT-dy -> Path to the file with the signed distance function determining in/out region.")
        print("        - pathToXCATDVFInOutSDT-dz -> Path to the file with the signed distance function determining in/out region.")
        print("        - outputDir                -> Path to where the output will be saved ")
        print("        - tmpOutputDir             -> Path to where the temporary output will be saved, choose a fast accessible disk space if available ")
        print("                                      Files in this folder will be deleted after processing completed.")
        print(" ")
        sys.exit()
    
    XCATdvfNiiFileNameIn          = sys.argv[1]
    XCATInOuySDTNiftiFileNameIn   = sys.argv[2] 
    XCATInOuySDTdxNiftiFileNameIn = sys.argv[3] 
    XCATInOuySDTdyNiftiFileNameIn = sys.argv[4] 
    XCATInOuySDTdzNiftiFileNameIn = sys.argv[5] 
    outputDirectoryName           = sys.argv[6]
    
    if len( sys.argv ) > 7:
        tempOutputDirectoryName  = sys.argv[7]
    else: 
        tempOutputDirectoryName = None
    
    if len( sys.argv ) > 8:
        maxNumOfOMPProcessors = int(sys.argv[8])
    else:
        maxNumOfOMPProcessors = None
    
    # Set the path to the nifty-reg and c3d executables here if they are not on the path
    niftyRegBinDir = ''
    c3dBinDir = ''
    
    if findExecutable.findExecutable( niftyRegBinDir + 'reg_resample' ) is None:
        print( 'Cannot find reg_resample in path. Please update niftyRegBinDir ')
        sys.exit()

    if findExecutable.findExecutable( niftyRegBinDir + 'reg_transform' ) is None:
        print( 'Cannot find reg_transform in path. Please update niftyRegBinDir ')
        sys.exit()
        
    XCATdvfPostProcessing( XCATdvfNiiFileNameIn, 
                           XCATInOuySDTNiftiFileNameIn, 
                           XCATInOuySDTdxNiftiFileNameIn, XCATInOuySDTdyNiftiFileNameIn, XCATInOuySDTdzNiftiFileNameIn,
                           outputDirectoryName, 
                           tempOutputDirectoryName,
                           maxNumOfOMPProcessors, 
                           niftyRegBinDirIn=niftyRegBinDir )
