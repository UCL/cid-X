#!/usr/bin/env python3



# Add the current directory to the system path before continuing
import sys
import os
sys.path.append( os.path.dirname( __file__ ) )

from glob import glob
from scipy.ndimage import label
import numpy as np
import configparser as cfp
import nibabel as nib
import XCATdvfPreProcessing as xPre
import XCATdvfPostProcessing as xPost
import convertXCATDVFTextFile
import commandExecution as cmdEx
import re



# Main entry point to post-processing framework
if __name__ == '__main__':
    print( "Tool to post-process XCAT DVF text files." )
    
    if len(sys.argv) < 2:
        print("Usage: XCATdvfProcessing.py pathToConfigFile")
        print("        - pathToConfigFile     -> Path to the configuration file that contains all parameters for processing")
        print(" ")
        
        print(" ")
        sys.exit()
    
    # Read the config file 
    configFileNameIn = sys.argv[1]
    parser = cfp.ConfigParser()

    try:
        parser.read( configFileNameIn )
    except:
        print("Could not read configuration file")
        sys.exit(1)

    
    # Determine the processing steps that need to be performed
    doPreProcessing       = False
    doBatchPostProcessing = False
    doBatchWarping        = False

    if 'doPreProcessing' in parser[ 'PROCESSING_STEPS' ].keys() :
        if ( ( parser[ 'PROCESSING_STEPS' ][ 'doPreProcessing' ].lower() != '0'     ) and 
             ( parser[ 'PROCESSING_STEPS' ][ 'doPreProcessing' ].lower() != 'false' ) and 
             ( parser[ 'PROCESSING_STEPS' ][ 'doPreProcessing' ].lower() != 'no'    ) ):
            doPreProcessing = True

    if 'doBatchPostProcessing' in parser[ 'PROCESSING_STEPS' ].keys() :
        if ( ( parser[ 'PROCESSING_STEPS' ][ 'doBatchPostProcessing' ].lower() != '0'     ) and 
             ( parser[ 'PROCESSING_STEPS' ][ 'doBatchPostProcessing' ].lower() != 'false' ) and 
             ( parser[ 'PROCESSING_STEPS' ][ 'doBatchPostProcessing' ].lower() != 'no'    ) ):
            doBatchPostProcessing = True
    
    if 'doBatchWarping' in parser[ 'PROCESSING_STEPS' ].keys() :
        if ( ( parser[ 'PROCESSING_STEPS' ][ 'doBatchWarping' ].lower() != '0'     ) and 
             ( parser[ 'PROCESSING_STEPS' ][ 'doBatchWarping' ].lower() != 'false' ) and 
             ( parser[ 'PROCESSING_STEPS' ][ 'doBatchWarping' ].lower() != 'no'    ) ):
            doBatchWarping = True
    
    
    if doPreProcessing:
        try:
            # Run the pre-processing
            preProcessor = xPre.XCATdvfPreProcessing()
            preProcessor.configureByParser( parser )
            preProcessor.run()
        except:
            print("ERROR: Pre-processing failed")
            sys.exit(1)
    

    if doBatchPostProcessing:
        # Collect parameters for batch conversion
        try:
            dvfTextFilePattern = parser['BATCH_POSTPROCESSING']['xcatDVFFilePattern']
            outputDirectory    = parser['BATCH_POSTPROCESSING']['outDir']
            corruptedFilesDir  = parser['BATCH_POSTPROCESSING']['corruptedFilesDir']
            
            
            # Retrieve data from pre-processing section
            imageSize    = np.array([0,0,0], dtype=np.int   )
            imageSize[0] = parser['PREPROCESSING']['numVoxX']
            imageSize[1] = parser['PREPROCESSING']['numVoxY']
            imageSize[2] = parser['PREPROCESSING']['numVoxZ']

            imageSpacing = np.array([0,0,0], dtype=np.float )
            imageSpacing[0] = parser['PREPROCESSING']['spacingX']
            imageSpacing[1] = parser['PREPROCESSING']['spacingY']
            imageSpacing[2] = parser['PREPROCESSING']['spacingZ']

            sdt_1_fileName   = os.path.join( parser['PREPROCESSING']['outDir'], parser['PREPROCESSING']['outDistMapImgName'] )
            sdtDx_1_fileName = os.path.join( parser['PREPROCESSING']['outDir'], parser['BATCH_POSTPROCESSING']['outDistMapDxImgName']  )
            sdtDy_1_fileName = os.path.join( parser['PREPROCESSING']['outDir'], parser['BATCH_POSTPROCESSING']['outDistMapDyImgName']  )
            sdtDz_1_fileName = os.path.join( parser['PREPROCESSING']['outDir'], parser['BATCH_POSTPROCESSING']['outDistMapDzImgName']  )

            xcatAtnImgFileName      = os.path.join( parser['PREPROCESSING']['outDir'], parser['PREPROCESSING']['outXCATAtnImgName'] )
            tmpDir                  = parser['BATCH_POSTPROCESSING']['tmpDir']
            numberOfProcessorsToUse = int( parser['BATCH_POSTPROCESSING']['numProcessorsDVFInv'] )
            
            niftyRegBinDir = ''
            if 'niftyRegBinDir' in parser[ 'BATCH_POSTPROCESSING' ].keys() :
                niftyRegBinDir = parser[ 'BATCH_POSTPROCESSING' ][ 'niftyRegBinDir' ]
        
        except:
            print('ERROR: Batch processing failed')
            sys.exit(1)
        
        txtFileList = glob( dvfTextFilePattern )
    
        for txtDVFFile in txtFileList:
            
            txtDVFFile = txtDVFFile.replace( '\\', '/' )
            
            try:
                # Generate the nifti DVF
                niftiFileNames = convertXCATDVFTextFile.convertXCATDVFTextFileToNiftiImage( txtDVFFile, 
                                                                                            xcatAtnImgFileName, 
                                                                                            outputDirectory, 
                                                                                            imageSize, imageSpacing, 
                                                                                            False, False )
                
                # Generate the corrected forward and backward DVF
                xPost.XCATdvfPostProcessing( niftiFileNames['dvfImageFileName'], 
                                             sdt_1_fileName, 
                                             sdtDx_1_fileName, sdtDy_1_fileName, sdtDz_1_fileName, 
                                             outputDirectory, 
                                             tmpDir, numberOfProcessorsToUse, 
                                             niftyRegBinDir )
            
            except:
                # Write the DVF to the corrupted files directory
                os.makedirs( corruptedFilesDir, exist_ok=True  )
                os.rename( txtDVFFile, corruptedFilesDir + os.path.basename( txtDVFFile ) )
                continue

        
    if doBatchWarping:
        try:
            doLungCTIntensityScaling = False
            # Need to prepare the intensity scaling if required
            if 'scaleLungIntensity' in parser[ 'BATCH_WARPING' ].keys() :
                if ( ( parser[ 'BATCH_WARPING' ][ 'scaleLungIntensity' ].lower() != '0'     ) and 
                     ( parser[ 'BATCH_WARPING' ][ 'scaleLungIntensity' ].lower() != 'false' ) and 
                     ( parser[ 'BATCH_WARPING' ][ 'scaleLungIntensity' ].lower() != 'no'    ) ):
                    doLungCTIntensityScaling = True
            
            # Check which nifty-reg version to use
            niftyRegBinDir = ''
            if 'niftyRegBinDir' in parser[ 'BATCH_POSTPROCESSING' ].keys() :
                niftyRegBinDir = parser[ 'BATCH_POSTPROCESSING' ][ 'niftyRegBinDir' ]
            
            # Use the output directory of the previous step
            dvfDir               = parser['BATCH_POSTPROCESSING']['outDir']

            dvfPostFix           = parser['BATCH_WARPING']['dvfPostFix']
            warpedOutDir         = parser['BATCH_WARPING']['outDir']
            warpedOutBaseImgName = parser['BATCH_WARPING']['warpedOutImgBaseName']

            if 'additionalResampleParams' in parser['BATCH_WARPING'].keys():
                additionalResampleParams = parser['BATCH_WARPING']['additionalResampleParams']
            else:
                additionalResampleParams = ''

            if 'referenceImgName' in parser['BATCH_WARPING'].keys():
                referenceImgName = parser['BATCH_WARPING']['referenceImgName']
            else:
                referenceImgName = os.path.join( parser['PREPROCESSING']['outDir'], parser['PREPROCESSING']['outXCATAtnImgName']  )

            os.makedirs( warpedOutDir, exist_ok=True )
            
            # First generate a lung mask
            if doLungCTIntensityScaling:
                # Load the image with the intensities to be scaled
                ctNiiImage = nib.load( referenceImgName )
                
                # Get the upper and lower threshold
                lungLowerThreshold = float( parser['BATCH_WARPING']['lungLowerThreshold'] )
                lungUpperThreshold = float( parser['BATCH_WARPING']['lungUpperThreshold'] )
                
                # Perform the thresholding
                imgData = ctNiiImage.get_data()
                lungMask = (imgData > lungLowerThreshold) & (imgData < lungUpperThreshold)
                
                # find the two largest connected components by first labelling the image data
                lungLabels, numLabels = label( lungMask )
                
                # then find the two largest labels
                labelCount = np.zeros([numLabels+1,2])
                
                for i in range( 1, numLabels+1 ):
                    labelCount[i,1] = np.sum( lungLabels == i )
                    labelCount[i,0] = i
                
                lungMask[:, :, :] = 0
                
                for i in range(2):
                    maxCountIDX = np.where( labelCount[:,1]  == np.max(labelCount[:,1]))
                    maxRegionVal = labelCount[:,0][maxCountIDX]
                    
                    lungMask[lungLabels==maxRegionVal] = 1
                    
                    labelCount[:,1][maxCountIDX] = 0
                
                lungMaskNii = nib.Nifti1Image( np.array( lungMask , dtype=np.uint8), ctNiiImage.affine )
                nib.save(lungMaskNii, warpedOutDir + 'lungMask.nii.gz')

            # Find all the files that match the defined pattern
            # Globbing does not provide a correctly sorted list, furthermore it does not guarantee the order to be the
            # same if two separate lists are generated. Hence here the inverse DVF file name is generated without
            # globbing. File existence needs to be checked below.
            dvfFileList = glob(dvfDir + '*' + dvfPostFix)
            dvfFileList = sorted(dvfFileList, key=lambda e: int(re.findall('\d+',os.path.split(e)[-1].split(dvfPostFix)[0])[-1]) )

            for curDVFFileName_Nto1 in dvfFileList:
                
                # Extract the current file number from the file name using the given postfix
                curNumber = int( curDVFFileName_Nto1.split('_to_frame')[1].split(dvfPostFix)[0] )
                curOutputFileName = warpedOutDir + warpedOutBaseImgName + '_%04i.nii.gz' % curNumber
                curScaledCTFileName = warpedOutDir + 'curScaled_%04i.nii.gz' % curNumber
                curJacImgFile = warpedOutDir + 'jac%04i.nii.gz' % curNumber

                if doLungCTIntensityScaling:
                    # Check that the inverse DVF exists
                    curDVFFileName_1toN = curDVFFileName_Nto1[0:curDVFFileName_Nto1.rfind('Nto1')] + '1toN' + curDVFFileName_Nto1[curDVFFileName_Nto1.rfind('Nto1') + 4:]

                    if not os.path.exists( curDVFFileName_1toN ):
                        print( "Error: Expected the DVF file {}, but could not find it here.".format(curDVFFileName_1toN) )
                        print( "Cannot perform intensity scaling at this time point." )
                        continue

                    # Calculate the Jacobian map if using
                    jacobianCMD = niftyRegBinDir + 'reg_jacobian'
                    jacobianParams = ' -trans ' + curDVFFileName_1toN
                    jacobianParams += ' -jac ' + curJacImgFile
                    
                    cmdEx.runCommand( jacobianCMD, jacobianParams, logFileName=warpedOutDir + 'jacobianLog.txt',
                                      workDir=warpedOutDir, onlyPrintCommand=False )
                    
                    # Change the intensities of the reference image name
                    curJacNii = nib.load( curJacImgFile )
                    
                    scaling = np.ones_like( ctNiiImage.get_data() )
                    scaling[ lungMask ] = curJacNii.get_data()[ lungMask ] 
                    
                    # Limit unrealistic scaling
                    scaling[scaling < 0.5] = 0.5
                    scaling[scaling > 1.5] = 1.5                    
                    
                    scaledCTData = (ctNiiImage.get_data() + 1000.0 ) / scaling - 1000.0
                    scaledCTNii = nib.Nifti1Image(scaledCTData, ctNiiImage.affine)
                    nib.save( scaledCTNii, curScaledCTFileName )
                    
                    referenceImgName = curScaledCTFileName
                    
                # Construct the command and run the resampleing 
                resampleCMD     = niftyRegBinDir + 'reg_resample'
                resampleParams  = ' -ref '   + referenceImgName
                resampleParams += ' -flo '   + referenceImgName
                resampleParams += ' -res '   + curOutputFileName
                resampleParams += ' -trans ' + curDVFFileName_Nto1
                resampleParams += ' '        + additionalResampleParams
    
                cmdEx.runCommand( resampleCMD, resampleParams, logFileName=warpedOutDir+'warpingLog.txt',
                                  workDir=warpedOutDir, onlyPrintCommand=False)
                
                # Clean up the intermediate files
                if os.path.exists( curScaledCTFileName ):
                    os.remove(curScaledCTFileName )
                if os.path.exists( curJacImgFile ):
                    os.remove(curJacImgFile )
                
        except: 
            print('ERROR: Batch warping of reference image failed')
            sys.exit(1)
