

'''
@author:  Bjoern Eiben
@summary: Implements a class which allows to detect and correct folding of displacement vector fields
'''

import finiteDifferences as finiteDif
import numpy as np
from scipy import ndimage


class DisplacementVectorFieldTools3D( finiteDif.finiteDifferences3D ):
    '''
    @summary: Class to perform basic operations on 3D displacement vector fields (DVFs) given on a regular grid.
              These include DVF inversion, folding detection and correction. 
    '''
    
    def __init__( self, dx, dy, dz ):
        '''
            @param dx: Grid spacing of the displacement vector field in x-direction.
            @param dy: Grid spacing of the displacement vector field in y-direction.
            @param dz: Grid spacing of the displacement vector field in z-direction.
            @note: Change the parameter self.minAllowedJacobian to increase folding sensitivity.
        '''
        finiteDif.finiteDifferences3D.__init__( self, dx, dy, dz )
        self.minAllowedJacobian = 0.05
        self.smoothingRadius = 1.0
        self.verbose = False
        
        
        
        
    def checkForFolding( self, ux, uy, uz ):
        '''
            @summary: Analyses the given DVF and checks if folding occurred.
            @param ux: x-component of the DVF
            @param uy: y-component of the DVF
            @param uz: z-component of the DVF
        '''
        # Test for folding by using the deformation gradient and its determinant
        F11 = np.zeros_like(ux)
        F21 = np.zeros_like(ux)
        F31 = np.zeros_like(ux)

        F12 = np.zeros_like(ux)
        F22 = np.zeros_like(ux)
        F32 = np.zeros_like(ux)

        F13 = np.zeros_like(ux)
        F23 = np.zeros_like(ux)
        F33 = np.zeros_like(ux)

        self.diffX( ux, F11 )
        self.diffX( uy, F21 )
        self.diffX( uz, F31 )

        self.diffY( ux, F12 )
        self.diffY( uy, F22 )
        self.diffY( uz, F32 )

        self.diffZ( ux, F13 )
        self.diffZ( uy, F23 )
        self.diffZ( uz, F33 )
        
        # Add 1.0 to main diagonal as: F = I + grad(u)
        F11 = 1.0 + F11
        F22 = 1.0 + F22
        F33 = 1.0 + F33
        
        # Derive the Jacobian from this
        subDet1 = (F22 * F33 - F23 * F32)
        subDet2 = (F21 * F33 - F23 * F31)
        subDet3 = (F21 * F32 - F22 * F31)
        
        J = (   F11 * subDet1 
              - F12 * subDet2
              + F13 * subDet3 ) 
        
        foldingElements = np.where( J<=self.minAllowedJacobian )
        folding = foldingElements[0].shape[0] > 0 
        
        if folding and self.verbose:
            print('Detected folding: min(J) = %e, numElements invovled: %i' %(J.min(), foldingElements[0].shape[0]) )
            
        return folding, foldingElements




    def correctFolding( self, ux, uy, uz, enforceZeroDisplacementBoundary=True ):
        
        self.correct3dNAN( ux, 2 )
        self.correct3dNAN( uy, 2 )
        self.correct3dNAN( uz, 2 )
        
        [folding, foldingElements] = self.checkForFolding( ux, uy, uz )
        numFoldingCorrectionSteps = 0
        
        while folding:
            numFoldingCorrectionSteps += 1
            #
            # Correct folding
            # Idea: Locally smooth the DVF until no more folding occurs    
            #
            foldingMask = np.zeros_like( ux )
            foldingMask[foldingElements] = 1.0 
            
            smoothFoldingMask = ndimage.gaussian_filter( foldingMask, self.smoothingRadius )
            
            uxUpSmooth = ndimage.gaussian_filter( ux, 1.0 )
            uyUpSmooth = ndimage.gaussian_filter( uy, 1.0 )
            uzUpSmooth = ndimage.gaussian_filter( uz, 1.0 )
            
            if enforceZeroDisplacementBoundary:
                self._setBorderToZeroDisplacement( uxUpSmooth )
                self._setBorderToZeroDisplacement( uyUpSmooth )
                self._setBorderToZeroDisplacement( uzUpSmooth )
            
            uxNew = ux * (1.0 - smoothFoldingMask) + uxUpSmooth * (smoothFoldingMask)
            uyNew = uy * (1.0 - smoothFoldingMask) + uyUpSmooth * (smoothFoldingMask)
            uzNew = uz * (1.0 - smoothFoldingMask) + uzUpSmooth * (smoothFoldingMask)
            
            # check again for folding
            folding, foldingElements = self.checkForFolding( uxNew, uyNew, uzNew )
            
            ux = uxNew
            uy = uyNew
            uz = uzNew
        
        if self.verbose:
            print('Folding DVF was corrected in %i iterations.' % numFoldingCorrectionSteps )
        
        return ux, uy, uz



    def _setBorderToZeroDisplacement( self, arrayIn ):
        arrayIn[:,:, 0] = 0.0
        arrayIn[:,:,-1] = 0.0
        arrayIn[:, 0,:] = 0.0
        arrayIn[:,-1,:] = 0.0
        arrayIn[ 0,:,:] = 0.0
        arrayIn[-1,:,:] = 0.0
        
        
        
        
    def correct3dNAN( self, arrayIn, minSummation = 2 ):
        '''
            @summary: Remove NaN elements from an array by interpolating by the closest neighbours. 
            @param arrayIn: The array which will be corrected
            @param minSummation: The minimum number of neighbours that have to be different from NaN before a summation is being performed.
                                 Valid options are 1, 2 and 3. If set higher, the algorithm will only converge in very limited scenarios.  
        '''
        
        idxNaN = np.where( np.isnan( arrayIn ) )
        
        max_x = arrayIn.shape[0]-1
        max_y = arrayIn.shape[1]-1
        max_z = arrayIn.shape[2]-1
        
        aCorrected = arrayIn.copy()
        
        while np.max(np.isnan(arrayIn)):
            idxNaN = np.where(np.isnan(arrayIn))
            
            for i in range( idxNaN[0].shape[0] ):
                
                curIDX_x = idxNaN[0][i]
                curIDX_y = idxNaN[1][i]
                curIDX_z = idxNaN[2][i]
                
                curIDX_x_low = np.clip( curIDX_x-1, 0, max_x )
                curIDX_y_low = np.clip( curIDX_y-1, 0, max_y )
                curIDX_z_low = np.clip( curIDX_z-1, 0, max_z )
        
                curIDX_x_high = np.clip( curIDX_x+1, 0, max_x )
                curIDX_y_high = np.clip( curIDX_y+1, 0, max_y )
                curIDX_z_high = np.clip( curIDX_z+1, 0, max_z )
                
                # compose indices of six nearest neighbours
                idxNN1 = ( curIDX_x_low,  curIDX_y,      curIDX_z      )
                idxNN2 = ( curIDX_x_high, curIDX_y,      curIDX_z      )
                idxNN3 = ( curIDX_x,      curIDX_y_low,  curIDX_z      ) 
                idxNN4 = ( curIDX_x,      curIDX_y_high, curIDX_z      )
                idxNN5 = ( curIDX_x,      curIDX_y,      curIDX_z_low  )
                idxNN6 = ( curIDX_x,      curIDX_y,      curIDX_z_high )
        
                idxNNs = [ idxNN1, idxNN2, idxNN3, idxNN4, idxNN5, idxNN6 ]
        
                numSums = 0
                valSum  = 0.0
                
                for idxNN in idxNNs:
                    valNN = arrayIn[idxNN]
                
                    if not np.isnan( valNN ):
                        numSums += 1
                        valSum += valNN
                
                if numSums >= minSummation:
                    aCorrected[ curIDX_x, curIDX_y, curIDX_z ] = valSum / (1.0 * numSums)
                    
            arrayIn[:,:] = aCorrected[:,:]
            
        del aCorrected

        
