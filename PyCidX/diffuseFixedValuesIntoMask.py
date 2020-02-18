

""" Use a diffusion process to smooth values into a region where a given mask is zero. 

Currently only implemented in 3D.  

"""
import finiteDifferences as fd
import numpy as np



class diffuseFixedValuesIntoMask(object):
    
    def __init__( self, fixedValuesIn, maskRegionIn, spacingIn, nTimeStepsIn ):
        self.fixedValues = fixedValuesIn
        self.maskRegion  = maskRegionIn
        self.spacing     = spacingIn
        self.nTimeSteps  = nTimeStepsIn
        self.finDif3D    = fd.finiteDifferences3D( self.spacing[0], self.spacing[0], self.spacing[0], 'central' )
        self.diffCoeff   = 0.35
        self.dT          = 0.5
    
    

    
    def diffuse(self):
        
        # Initialise the result with a zero matrix of same size as well as the gradients
        res   = np.zeros_like( self.fixedValues )
        resDxx = np.zeros_like( self.fixedValues )
        resDyy = np.zeros_like( self.fixedValues )
        resDzz = np.zeros_like( self.fixedValues )
        
        # Set the values inside the region to the original ones
        res[self.maskRegion != 0] = self.fixedValues[self.maskRegion != 0]
        
        # Perform the time tesps
        for i in range(self.nTimeSteps):
            if ( i % 10 ) == 0:
                print("diffusion iteration %i" %i) 
            # Note: This only works for constant isotropic diffusion
            # Calculate the second derivatives
            self.finDif3D.diffXX( res, resDxx ) 
            self.finDif3D.diffYY( res, resDyy )
            self.finDif3D.diffZZ( res, resDzz )
            
            # Do the time step
            res = res + self.dT * self.diffCoeff * (resDxx + resDyy + resDzz)
            
            # Fix the values to the input ones
            res[self.maskRegion != 0] = self.fixedValues[self.maskRegion != 0]
            
        return res
        

