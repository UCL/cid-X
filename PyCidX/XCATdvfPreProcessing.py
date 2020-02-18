#!/usr/bin/env python3

# Purpose: Generate the required inputs to batch-process the XCAT DVF text files.
#          In puts required here: 
#           1) binary file with the XCAT attenuation
#           2) first DVF text file

import sys
import numpy as np
import os
import convertXCATBinaryFile as cxbf
import convertXCATDVFTextFile as cxdt
import levelSetEvolution as lse
import signedDistanceMap as sdt
import nibabel as nib
import configparser as cfp




class XCATdvfPreProcessing( object ):
    
    
    def __init__( self, 
                  xcatAtnFile    = None, 
                  xcatDVFFile    = None, 
                  outDir         = None, 
                  imageDimension = None, 
                  voxelSpacing   = None ):
        
        # Essential parameters to be defined to run the pre-processing
        self.xcatAtnFile         = xcatAtnFile
        self.xcatDVFFile         = xcatDVFFile
        self.outDir              = outDir
        self.imageDimension      = imageDimension
        self.voxelSpacing        = voxelSpacing

        self.outDistMapImgName   = 'distMap.nii.gz'
        self.outLevelSetImgName  = 'levelSetOut.nii.gz'
        self.outXCATAtnImgName   = 'xcatImg.nii.gz'
        self.outXCATCTImgName    = None
        # optional parameters
        self.saveLevelSetImage          = True
        self.numberOfLevelSetIterations = 5000
        



    def run( self ):
        print( " " )
        print( "  Converting with parameters:" )
        print( "  - XCAT .bin file:         " + self.xcatAtnFile )
        print( "  - XCAT 1st DVF text file: " + self.xcatDVFFile )
        print( "  - output directory:       " + self.outDir )
        print( "  - image dimension:        " + str( self.imageDimension ) )
        print( "  - voxel size:             " + str( self.voxelSpacing ) )
        print( " " )
    
        # Generate the output path if it does not exist
        if not os.path.exists( self.outDir ):
            try: 
                os.makedirs( self.outDir )
                print( "  ... Created output directory ...")
            except: 
                print( "  ... ERROR: Could not generate output directory, exiting." )
                sys.exit()
        pass
        
        # Convert the binary/structural image into nifti format
        atnNiiFileName = cxbf.convertXCATBinaryFile( self.xcatAtnFile, 
                                                     self.outDir, 
                                                     self.imageDimension, 
                                                     self.voxelSpacing, 
                                                     False, 
                                                     self.outXCATAtnImgName )
        
        if self.outXCATCTImgName is not None:
            cxbf.convertXCATBinaryFile( self.xcatAtnFile, 
                                        self.outDir, 
                                        self.imageDimension, 
                                        self.voxelSpacing, 
                                        True, 
                                        self.outXCATCTImgName )

        
        
        # Convert the first DVF text file to extract the label information
        # Here we do not actually require the DVF information but rather the labels
        # that are contained in the XCAT DVF text file
        txtConvOutFileNames = cxdt.convertXCATDVFTextFileToNiftiImage( self.xcatDVFFile, 
                                                                       atnNiiFileName, 
                                                                       self.outDir, 
                                                                       self.imageDimension, 
                                                                       self.voxelSpacing, 
                                                                       True, True, False )
        
        # Start the level-set evolution
        initialImage = nib.load( txtConvOutFileNames[ 'labelInImageFileName'  ] )
        speedImage   = nib.load( txtConvOutFileNames[ 'labelOutImageFileName' ] )
        
        levelSetter = lse.LevelSetEvolution( initialImage, speedImage )
        levelSetter.numberOfIterations = self.numberOfLevelSetIterations
        print( "  ... Running level-set evolution with %i iterations. This might take a while..." % levelSetter.numberOfIterations )
        levelSetter.run()
        
        # Save level-set image if requested
        if self.saveLevelSetImage:
            levelSetter.saveLSOutput( self.outDir + self.outLevelSetImgName )
        
        # Convert the level-set result into a full signed-distance map
        # Note: We could simply use the output of the level-sets, however, 
        #       Since the signed distance map needs to be calculated in the 
        #       DVF inversion procedure, for consistency we decided to update 
        #       the signed distance map instead.
        sdtMapper = sdt.SignedDistanceMap( levelSetter.outLSImage )
        sdtMapper.run()
        nib.save( sdtMapper.distMapImg, self.outDir + self.outDistMapImgName )
        
        return
    
    
    
    
    def configureByParser( self, parserIn ):

        try:
            # Set the inputs
            self.xcatAtnFile       = parserIn['PREPROCESSING']['xcatAtnFile']
            self.xcatDVFFile       = parserIn['PREPROCESSING']['xcatDVFFile']

            # and outputs
            self.outDir            = parserIn['PREPROCESSING']['outDir']
            self.outDistMapImgName = parserIn['PREPROCESSING']['outDistMapImgName']
            self.outXCATAtnImgName = parserIn['PREPROCESSING']['outXCATAtnImgName']
            if 'outXCATCTImgName' in parserIn['PREPROCESSING'].keys() :
                self.outXCATCTImgName =  parserIn[ 'PREPROCESSING' ][ 'outXCATCTImgName' ]

            self.imageDimension    = np.array( [0,0,0], dtype=np.int )
            self.voxelSpacing      = np.array( [0.0, 0.0, 0.0] )

            self.imageDimension[0] = int( parserIn['PREPROCESSING']['numVoxX'] )
            self.imageDimension[1] = int( parserIn['PREPROCESSING']['numVoxY'] )
            self.imageDimension[2] = int( parserIn['PREPROCESSING']['numVoxZ'] )
        
            self.voxelSpacing[0] = float( parserIn['PREPROCESSING']['spacingX'] )
            self.voxelSpacing[1] = float( parserIn['PREPROCESSING']['spacingY'] )
            self.voxelSpacing[2] = float( parserIn['PREPROCESSING']['spacingZ'] )

            # Set optional parameters
            if 'numberOfLevelSetIterations' in parserIn['PREPROCESSING'].keys() :
                self.numberOfLevelSetIterations =  int( parserIn[ 'PREPROCESSING' ][ 'numberOfLevelSetIterations' ] )

            if 'saveLevelSetImage' in parserIn[ 'PREPROCESSING' ].keys() :
                if ( ( parserIn[ 'PREPROCESSING' ][ 'saveLevelSetImage' ].lower() != '0'     ) and 
                     ( parserIn[ 'PREPROCESSING' ][ 'saveLevelSetImage' ].lower() != 'false' ) and 
                     ( parserIn[ 'PREPROCESSING' ][ 'saveLevelSetImage' ].lower() != 'no'    ) ):
                    self.saveLevelSetImage = True
                else:
                    self.saveLevelSetImage = False
                    
            if 'outDistMapImgName' in parserIn[ 'PREPROCESSING' ].keys() :
                self.outLevelSetImgName =  parserIn[ 'PREPROCESSING' ][ 'outLevelSetImgName' ]

        except:
            print(" Error: Configuration failed! ")
            print(" ")
            print(" Required (and optional) configuration contents:")
            print(" ")
            print("[PREPROCESSING]")
            print("xcatAtnFile       = /path/to/XCAT_binaryVolume.bin")
            print("xcatDVFFile       = /path/to/XCAT_dvf_text_file_vec_frame1_to_frameAny.txt")
            print("outDir            = /path/to/where/files/will/be/written/")
            print("outDistMapImgName = distMapImageFileName.nii.gz")
            print("outXCATAtnImgName = convertedXCATBinaryImage.nii.gz")
            print("numVoxX           = 123    # number of voxels in x-direction")
            print("numVoxY           = 123    # number of voxels in y-direction")
            print("numVoxZ           = 123    # number of voxels in z-direction")
            print("spacingX          = 1.23   # spacing in x-direction")
            print("spacingY          = 1.23   # spacing in x-direction")
            print("spacingZ          = 1.23   # spacing in x-direction")
            print(" ")
            print("# Optional ")
            print("numberOfLevelSetIterations = 123   # number of iterations of the LS evolution")
            print("outLevelSetImgName         = levelSetImageFileName.nii.gz.nii.gz ")
            print("saveLevelSetImage          = 0   # Can be 0, false, no to not save the image")

            sys.exit(1)
        
            



if __name__ == "__main__":
    
    print( "Tool to generate the inputs for the post-processing of XCAT DVF text files." )
    
    if len(sys.argv) < 2:
        print("Usage: XCATdvfPostProcessing.py pathToConfigFile")
        print("        - pathToConfigFile     -> Path to the configuration file that contains all parameters for pre-processing")
        print(" ")
        
        print(" ")
        sys.exit()
    
    # Read the config file 
    configFileNameIn = sys.argv[1]
    
    parser = cfp.ConfigParser()
    
    # Read the config file
    try:
        parser.read( configFileNameIn )

    except:
        print(" Required configuration file contents:")
        print(" ")
        print("[PREPROCESSING]")
        print("xcatAtnFile       = /path/to/XCAT_binaryVolume.bin")
        print("xcatDVFFile       = /path/to/XCAT_dvf_text_file_vec_frame1_to_frameAny.txt")
        print("outDir            = /path/to/where/files/will/be/written/")
        print("outDistMapImgName = distMapImageFileName.nii.gz")
        print("numVoxX           = 123    # number of voxels in x-direction")
        print("numVoxY           = 123    # number of voxels in y-direction")
        print("numVoxZ           = 123    # number of voxels in z-direction")
        print("spacingX          = 1.23   # spacing in x-direction")
        print("spacingY          = 1.23   # spacing in x-direction")
        print("spacingZ          = 1.23   # spacing in x-direction")
        print(" ")
        print("# Optional ")
        print("numberOfLevelSetIterations = 123   # number of iterations of the LS evolution")
        print("outLevelSetImgName         = levelSetImageFileName.nii.gz.nii.gz ")
        print("saveLevelSetImage          = 0   # Can be 0, false, no to not save the image")
        sys.exit(1)


    # Run the pre-processing
    preProcessor = XCATdvfPreProcessing()
    preProcessor.configureByParser( parser )
    preProcessor.run()
    
