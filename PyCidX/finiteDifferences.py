#!/usr/bin/env python3


'''
@author: Bjoern Eiben
@summary: Classes which collects the 2D (finiteDifferences2D) and 3D (finiteDifferences3D) finite difference operations.
'''


import numpy as np


class finiteDifferences2D:
    
    
    def __init__( self, dx=1.0, dy=1.0, differentiationScheme='central' ):
        ''' 
            @summary: Define the regular grid spacing in x (1st array axis) and y (2nd array axis) direction
            @note: When the parameters dx and dy are changed after initialisation, call .initialise() to 
                   update pre-calculated values!
                   Set .diffScheme to 'central', 'fwdBwd', 'fwd', 'bwd' or 'central4'
            @param dx: Grid spacing in the direction of the first array axis 
            @param dy: Grid spacing in the direction of the second array axis 
        '''
        self.dx = dx
        self.dy = dy

        self.diffScheme = differentiationScheme

        # Pre-calculate derived values
        self.initialise()
        
        
        
        
    def initialise( self ):
        ''' 
            @summary: Pre-calculate values derived from dx and dy
        '''
        
        #
        # Note: The multiplication is much more efficient than the division. Hence the reciprocal values are pre-calculated.
        #       The _oo_ means one-over 
        # 
        self._oo_dx    = 1.0 / self.dx
        self._oo_dy    = 1.0 / self.dy
         
        self._oo_2dx   = 1.0 / (2.0 * self.dx)
        self._oo_2dy   = 1.0 / (2.0 * self.dy)
        
        self._oo_dxdy  = 1.0 / (self.dx * self.dy)
        self._oo_2dxdy = 1.0 / (2.0 * self.dx * self.dy)
        self._oo_4dxdy = 1.0 / (4.0 * self.dx * self.dy)
        
        self._oo_12dx  = 1.0 / (12.0 * self.dx)
        self._oo_12dy  = 1.0 / (12.0 * self.dy)
        
        self._oo_dx2   = 1.0 / (self.dx * self.dx)
        self._oo_dy2   = 1.0 / (self.dy * self.dy)
        
    
    
    
    def diff( self, i, axis1, axis2=None ):
        ''' 
            @summary: Differentiation into a certain direction. Central differences are always used here. 
            @return: The differentiation result
            @param i: The input
            @param axis1: First differentiation into this direction
            @param axis2: Second differentiation direction. Set to None if first derivative is required only 
        '''
        o = np.zeros_like(i)
        
        if axis2 == None:
            # first derivative
            if axis1 == 0:
                self.diffXC(i, o)
            elif axis1 == 1:
                self.diffYC(i, o)
        
        # second/mixed derivative...
        elif axis2 == 0:
            if axis1 == 0:
                self.diffXX(i, o)
            elif axis1 == 1:
                self.diffXY(i, o)
        
        elif axis2 == 1:
            if axis1 == 0:
                self.diffXY(i, o)
            elif axis1 == 1:
                self.diffYY(i, o)
                
        return o
            
            
    
    def diffX( self, i, o, it=0 ):
        ''' 
            @summary: Differentiation in the X direction of a given input array according the the specified diffScheme.
            @note: The output is not generated. 
            @param i: Input array
            @param o: Output array 
        '''
        if self.diffScheme =='central':
            self.diffXC(i, o)
        elif self.diffScheme == 'fwd':
            self.diffXF(i, o)
        elif self.diffScheme == 'bwd':
            self.diffXB(i, o)
        elif self.diffScheme == 'central4':
            self.diffXC2(i, o)
        else :
            if (it % 2) == 0:
                self.diffXF(i, o)
                return 
            else:
                self.diffXB(i, o)
                return
            
            
            
            
    def diffY(self, i, o, it=0):
        ''' 
            @summary: Differentiation in the Y direction of a given input array according the the specified diffScheme.
            @note: The output is not generated. 
            @param i: Input array
            @param o: Output array 
        '''
        
        if self.diffScheme =='central':
            self.diffYC(i, o)
        elif self.diffScheme == 'fwd':
            self.diffYF(i, o)
        elif self.diffScheme == 'bwd':
            self.diffYB(i, o)
        elif self.diffScheme == 'central4':
            self.diffYC2(i, o)
        else :
            if (it % 2) == 0:
                self.diffYF(i, o)
                return 
            else:
                self.diffYB(i, o)
                return
            
            

        
    def diffXC( self, i, o ):
        ''' 
            @summary: Calculates the central difference of an input scalar field in the x-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''
        o[1:-1,:] = ( - i[ 0:-2, : ]   
                      + i[ 2:  , : ] )  * self._oo_2dx #/ (2.0*self.dx)
        o[   0,:] = ( - i[    0, : ] 
                      + i[    1, : ] ) * self._oo_dx  #/ (self.dx)
        o[  -1,:] = ( - i[   -2, : ] 
                      + i[   -1, : ] ) * self._oo_dx  #/ (self.dx)
                      
                      
                      
                      
    def diffYC( self, i, o ):
        ''' 
            @summary: Calculates the central difference of an input scalar field in the y-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function
        '''
        o[:,1:-1] = ( - i[ :, 0:-2 ]   
                      + i[ :, 2:   ] ) * self._oo_2dy #/ (2.0*self.dy)
        o[ :,  0] = ( - i[ :,    0 ] 
                      + i[ :,    1 ] ) * self._oo_dy #/ (self.dy)
        o[ :, -1] = ( - i[ :,   -2 ] 
                      + i[ :,   -1 ] ) * self._oo_dy #/ (self.dy)
                      
                      
                      
                      
    def diffXF( self, i, o ):
        ''' 
            @summary: Calculates the forward difference of an input scalar field in the x-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''
        # Forward difference used:
        # pos.    | x- | x0 | x+ |
        # coeff.  |  0 | -1 | +1 |
        #
        o[0:-1,:] = ( - i[ 0:-1, :]   
                      + i[ 1:  , :] ) * self._oo_dx #/ (self.dx)

        # Switch to backward difference only at upper end
        o[  -1,:] = ( - i[ -2, : ] 
                      + i[ -1, : ] ) * self._oo_dx #  / (self.dx)
                      
                      
                      
                      
    def diffYF( self, i, o ):
        ''' 
            @summary: Calculates the forward difference of an input scalar field in the y-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''
        o[:,0:-1] = ( - i[ :, 0:-1 ]   
                      + i[ :, 1:   ] ) * self._oo_dy #/ (self.dy)

        o[:,-1] = ( - i[ :, -2 ] 
                    + i[ :, -1 ] ) * self._oo_dy #/ (self.dy)
        
        
        
        
    def diffXB( self, i, o ):
        ''' 
            @summary: Calculates the backward difference of an input scalar field in the x-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''
        # Forward difference used:
        # pos.    | x- | x0 | x+ |
        # coeff.  | -1 | +1 |  0 |
        #
        o[1:,:]   = ( - i[ 0:-1, : ]   
                      + i[ 1:  , : ] ) * self._oo_dx #/ (self.dx)

        # Switch to forward difference only at lower end
        o[0,:] = ( - i[ 0, : ] 
                     + i[ 1, : ] ) * self._oo_dx #/ (self.dx)




    def diffYB( self, i, o ):
        ''' 
            @summary: Calculates the backward difference of an input scalar field in the y-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''
        o[:,1:]   = ( - i[ :, 0:-1 ]   
                      + i[ :, 1:   ] ) * self._oo_dy #/ (self.dy)

        # Switch to forward difference only at lower end
        o[:,0] = ( - i[ :, 0 ] 
                   + i[ :, 1 ] ) * self._oo_dy #/ (self.dy)




    def diffXC2( self, i, o ):
        ''' 
            @summary: Calculates the central difference of an input scalar field in the x-direction 
                      with 4th order accuracy using a wider kernel of size 5 and writes it into the 
                      output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''
        #
        # 4th order accuracy everywhere it is possible
        #
        #i[4:  ,:,:] * ( -1.0 /12.0 ) # -1/12 ->  -1  x+2 element
        #i[3:-1,:,:] * (  2.0 / 3.0 ) # +2/3  ->  +8  x+1 element
        #i[2:-2,:,:] * (  0.0       ) # 0     ->   0  x (central) element
        #i[1:-3,:,:] * ( -2.0 / 3.0 ) # -2/3  ->  -8  x-1 element
        #i[0:-4,:,:] * (  1.0 /12.0 ) # +1/12 ->  +1  x-2 element
        o[2:-2,:] = (  - 1.0 * i[4:  ,:]
                       + 8.0 * i[3:-1,:]
                       - 8.0 * i[1:-3,:]
                       + 1.0 * i[0:-4,:] ) * self._oo_12dx #/ (12.0 * self.dx)

        # simple central difference at borders...
        o[ 1,:] = ( - i[  0, : ]   
                    + i[  2, : ] ) * self._oo_2dx #/ (2.0*self.dx)
        o[-2,:] = ( - i[ -3, : ]   
                    + i[ -1, : ] ) * self._oo_2dx #/ (2.0*self.dx)
                      
        # and forward/backward difference everywhere else
        o[   0,:] = ( - i[    0, : ] 
                      + i[    1, : ] ) * self._oo_dx #/ (self.dx)
        o[  -1,:] = ( - i[   -2, : ] 
                      + i[   -1, : ] ) * self._oo_dx #/ (self.dx)
                        
                        
                        
                        
    def diffYC2( self, i, o ):
        ''' 
            @summary: Calculates the central difference of an input scalar field in the y-direction 
                      with 4th order accuracy using a wider kernel of size 5 and writes it into the 
                      output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''
        #
        # 4th order accuracy everywhere it is possible
        #
        o[:,2:-2] = (  - 1.0 * i[:,4:  ]
                       + 8.0 * i[:,3:-1]
                       - 8.0 * i[:,1:-3]
                       + 1.0 * i[:,0:-4] ) * self._oo_12dy#/ (12.0 * self.dy)

        # simple central difference at borders...
        o[:, 1] = ( - i[ :, 0]   
                    + i[ :, 2] ) * self._oo_2dy #/ (2.0*self.dy)
        o[:,-2] = ( - i[ :,-3]   
                    + i[ :,-1] ) * self._oo_2dy #/ (2.0*self.dy)
                      
        # and forward/backward difference everywhere else
        o[ :,  0] = ( - i[  :,  0 ] 
                      + i[  :,  1 ] ) * self._oo_dy #/ (self.dy)
        o[ :, -1] = ( - i[  :, -2 ] 
                      + i[  :, -1 ] ) * self._oo_dy #/ (self.dy)






    def diffXX( self, i, o ):
        ''' 
            @summary: Calculates the second difference (1, -2, 1) of an input scalar field in the x-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''
        # central        
        o[1:-1,:] =  (   1.0 * i[ 0:-2, : ]  
                       - 2.0 * i[ 1:-1, : ] 
                       + 1.0 * i[ 2:  , : ] ) * self._oo_dx2 #/ self.dx2
        
        # switch to forward backward which is equivalent to copying the data      
        o[ 0,:] = o[ 1,:]
        o[-1,:] = o[-2,:]
                        
                        
    
    
    def diffYY( self, i, o ):   
        ''' 
            @summary: Calculates the second difference (1, -2, 1) of an input scalar field in the y-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''     
        o[:,1:-1] =  (   1.0 * i[ :, 0:-2 ]  
                       - 2.0 * i[ :, 1:-1 ] 
                       + 1.0 * i[ :, 2:   ] ) * self._oo_dy2 #/ self.dy2
    
        # switch to forward backward which is equivalent to copying the data      
        o[ :,0] = o[:, 1]
        o[:,-1] = o[:,-2]
                        
    

    

    def diffXY( self, i, o ):
        ''' 
            @summary: Calculates the mixed discrete derivative difference of an input scalar field in the xy-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''
        o[1:-1, 1:-1 ] = (   i[ 2:  , 2:   ]  
                           + i[ 0:-2, 0:-2 ]  
                           - i[ 2:  , 0:-2 ] 
                           - i[ 0:-2, 2:   ] ) * self._oo_4dxdy #/ (4.0 * self.dxy) 
        
        o[0, 1:-1 ] = (   i[ 1, 2:   ]  
                        + i[ 0, 0:-2 ]  
                        - i[ 1, 0:-2 ] 
                        - i[ 0, 2:   ] ) * self._oo_2dxdy #/ (2.0*self.dxy) 
        
        o[-1, 1:-1 ] = (   i[ -1, 2:   ]  
                         + i[ -2, 0:-2 ]  
                         - i[ -1, 0:-2 ] 
                         - i[ -2, 2:   ] ) * self._oo_2dxdy #/ (2.0*self.dxy)
        
        o[1:-1, 0 ] = (   i[2:  , 1 ]  
                        + i[0:-2, 0 ]  
                        - i[0:-2, 1 ] 
                        - i[2:  , 0 ] ) * self._oo_2dxdy #/ (2.0*self.dxy)
        
        o[1:-1, -1 ] = (   i[ 2:  , -1 ]  
                         + i[ 0:-2, -2 ]  
                         - i[ 0:-2, -1 ] 
                         - i[ 2:  , -2 ] ) * self._oo_2dxdy #/ (2.0*self.dxy)
        
        o[0, 0 ] = (   i[ 1, 1 ]  
                     + i[ 0, 0 ]  
                     - i[ 1, 0 ] 
                      - i[ 0, 1 ] ) * self._oo_dxdy #/ (self.dxy)
        
        o[-1, -1 ] = (   i[-1,-1 ]  
                       + i[-2,-2 ]  
                       - i[-1,-2 ] 
                       - i[-2,-1 ] ) * self._oo_dxdy #/ (self.dxy)
        
        o[0, -1 ] = (   i[ 1,-1 ]  
                      + i[ 0,-2 ]  
                      - i[ 1,-2 ] 
                      - i[ 0,-1 ] ) * self._oo_dxdy #/ (self.dxy)

        o[-1, 0 ] = (   i[ -1,1 ]  
                      + i[ -2,0 ]  
                      - i[ -2,1 ] 
                      - i[ -1,0 ] ) * self._oo_dxdy #/ (self.dxy)
        
        
                         


    def avgXF( self, i, o ):
        ''' 
            @summary: Calculates the forward average of an input scalar field in the x-direction 
                      and writes it into the output field provided. This can be interpreted that
                      the value of the average is located at inter-node positions
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''
        # Forward average used:
        # pos.    | x- | x0 | x+ |
        # coeff.  |  0 | .5 | .5 |
        #
        o[0:-1,:] = (   i[ 0:-1, : ]   
                      + i[ 1:  , : ] ) * 0.5

        # Switch to backward average at upper end
        o[  -1,:] = (   i[ -2, : ] 
                      + i[ -1, : ] ) * 0.5

    
    
    
    def avgYF( self, i, o ):
        ''' 
            @summary: Calculates the forward average of an input scalar field in the y-direction 
                      and writes it into the output field provided. This can be interpreted that
                      the value of the average is located at inter-node positions
            @param i: Input scalar field (2D np.array)
            @param o: Output scalar field (2D np.array). Must have been allocated by calling function.
        '''

        o[0:-1,:] = (   i[ :, 0:-1 ]   
                      + i[ :, 1:   ] ) * 0.5

        # Switch to backward average at upper end
        o[  -1,:] = (   i[ :, -2 ] 
                      + i[ :, -1 ] ) * 0.5

        
        
                        
class finiteDifferences3D:
    
    
    
    def __init__( self, dx=1.0, dy=1.0, dz=1.0, differentiationScheme='central' ):
        ''' Define the regular grid spacing in x (1st array axis) and y (2nd array axis) direction
        
            :note: When the parameters ``dx``, ``dy`` or ``dz`` are changed after initialisation, call .initialise() to 
                   update pre-calculated values!
                   Set .diffScheme to ``central``, ``fwdBwd``, ``fwd``, ``bwd`` or ``central4``
            :param dx: Grid spacing in the direction of the first array axis. 
            :type  dx: float 
            :param dy: Grid spacing in the direction of the second array axis. 
            :type  dy: float 
            :param dz: Grid spacing in the direction of the third array axis.
            :type  dz: float 
        '''
        
        self.dx = dx
        self.dy = dy
        self.dz = dz
        
        self.diffScheme = differentiationScheme
        
        # Pre-calculate derived values
        self.initialise()
        
        
        
        
    def resetSpacing(self, dx, dy, dz):
        ''' Reset the spacing of the finite difference class
        
            :param dx: Spacing along the first coordinate axis
            :type  dx: float
            :param dy: Spacing along the second coordinate axis
            :type  dy: float
            :param dz: Spacing along the third coordinate axis
            :type  dz: float
        '''
        
        self.dx = dx
        self.dy = dy
        self.dz = dz
        
        self.initialise()
        
        
        
        
    def initialise( self ):
        ''' Pre-calculate values derived from dx and dy
        '''
        
        #
        # Note: The multiplication is much more efficient than the division. Hence the reciprocal values are pre-calculated.
        #       The _oo_ means one-over 
        # 
        self._oo_dx    = 1.0 / self.dx
        self._oo_dy    = 1.0 / self.dy
        self._oo_dz    = 1.0 / self.dz
         
        self._oo_2dx   = 1.0 / (2.0 * self.dx)
        self._oo_2dy   = 1.0 / (2.0 * self.dy)
        self._oo_2dz   = 1.0 / (2.0 * self.dz)
        
        self._oo_dxdy  = 1.0 / (self.dx * self.dy)
        self._oo_dxdz  = 1.0 / (self.dx * self.dz)
        self._oo_dydz  = 1.0 / (self.dy * self.dz)
        
        self._oo_2dxdy = 1.0 / (2.0 * self.dx * self.dy)
        self._oo_2dxdz = 1.0 / (2.0 * self.dx * self.dz)
        self._oo_2dydz = 1.0 / (2.0 * self.dy * self.dz)
        
        self._oo_4dxdy = 1.0 / (4.0 * self.dx * self.dy)
        self._oo_4dxdz = 1.0 / (4.0 * self.dx * self.dz)
        self._oo_4dydz = 1.0 / (4.0 * self.dy * self.dz)
        
        self._oo_12dx  = 1.0 / (12.0 * self.dx)
        self._oo_12dy  = 1.0 / (12.0 * self.dy)
        self._oo_12dz  = 1.0 / (12.0 * self.dz) # TODO: Check if used at all in 3D
        
        self._oo_dx2   = 1.0 / (self.dx * self.dx)
        self._oo_dy2   = 1.0 / (self.dy * self.dy)
        self._oo_dz2   = 1.0 / (self.dz * self.dz)
        
        
        
    def diff( self, i, axis1, axis2=None ):
        ''' Differentiation into a certain direction. Central differences are always used here.
         
            :param i: The input array
            :type  i: np.array
            :param axis1: First differentiation into this direction. Must be :math:`\\in \\{ 0,1,2\\}`.
            :type  axis1: int
            :param axis2: Second differentiation direction. Set to None if first derivative is required only 
            :return: The differentiation result
        '''
        o = np.zeros_like(i)
        
        if axis2 == None:
            # first derivative
            if axis1 == 0:
                self.diffXC(i, o)
            elif axis1 == 1:
                self.diffYC(i, o)
            elif axis1 == 2:
                self.diffZC(i, o)
        
        # second/mixed derivative...
        elif axis2 == 0:
            if axis1 == 0:
                self.diffXX(i, o)
            elif axis1 == 1:
                self.diffXY(i, o)
            elif axis1 == 2:
                self.diffXZ(i, o)
         
        elif axis2 == 1:
            if axis1 == 0:
                self.diffXY(i, o)
            elif axis1 == 1:
                self.diffYY(i, o)
            elif axis1 == 2:
                self.diffYZ(i, o)
 
        elif axis2 == 2:
            if axis1 == 0:
                self.diffXZ(i, o)
            elif axis1 == 1:
                self.diffYZ(i, o)
            elif axis1 == 2:
                self.diffZZ(i, o)
        
        return o


        
        
        
        
    def diffX( self, i, o, it=0 ):
        ''' 
            @summary: Differentiation in the X direction of a given input array according the the specified diffScheme.
            @note: The output is not generated. 
            @param i: Input array
            @param o: Output array 
        '''
        if self.diffScheme =='central':
            self.diffXC(i, o)
        elif self.diffScheme == 'fwd':
            self.diffXF(i, o)
        elif self.diffScheme == 'bwd':
            self.diffXB(i, o)
        elif self.diffScheme == 'central4':
            self.diffXC2(i, o)
        else :
            if (it % 2) == 0:
                self.diffXF(i, o)
                return 
            else:
                self.diffXB(i, o)
                return
            
            
            
            
    def diffY(self, i, o, it=0):
        ''' 
            @summary: Differentiation in the Y direction of a given input array according the the specified diffScheme.
            @note: The output is not generated. 
            @param i: Input array
            @param o: Output array 
        '''
        
        if self.diffScheme =='central':
            self.diffYC(i, o)
        elif self.diffScheme == 'fwd':
            self.diffYF(i, o)
        elif self.diffScheme == 'bwd':
            self.diffYB(i, o)
        elif self.diffScheme == 'central4':
            self.diffYC2(i, o)
        else :
            if (it % 2) == 0:
                self.diffYF(i, o)
                return 
            else:
                self.diffYB(i, o)
                return
            

        
        
    def diffZ(self, i, o, it=0):
        ''' 
            @summary: Differentiation in the Z direction of a given input array according the the specified diffScheme. 
            @note: The output is not generated. 
            @param i: Input array
            @param o: Output array 
        '''
        
        if self.diffScheme == 'central':
            self.diffZC(i, o)
        elif self.diffScheme == 'fwd':
            self.diffZF(i, o)
        elif self.diffScheme == 'bwd':
            self.diffZB(i, o)
        elif self.diffScheme == 'central4':
            self.diffZC2(i, o)
        else :
            if (it % 2) == 0:
                self.diffZF(i, o)
                return 
            else:
                self.diffZB(i, o)
                return

        
                
        
    def diffXC( self, i, o ):
        ''' 
            @summary: Calculates the central difference of an input scalar field in the x-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        o[1:-1,:,:] = ( - i[ 0:-2, :, : ]   
                        + i[ 2:  , :, : ] ) * self._oo_2dx
        o[   0,:,:] = ( - i[    0, :, : ] 
                        + i[    1, :, : ] ) * self._oo_dx
        o[  -1,:,:] = ( - i[   -2, :, : ] 
                        + i[   -1, :, : ] ) * self._oo_dx
               
               
               
               
    def diffYC( self, i, o ):
        ''' 
            @summary: Calculates the central difference of an input scalar field in the y-direction 
                      and writes it into the output field provided.
            @param i: Input scala field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function
        '''
        o[:,1:-1,:] = ( - i[ :, 0:-2, : ]   
                        + i[ :, 2:  , : ] ) * self._oo_2dy
        o[ :,  0,:] = ( - i[ :,    0, : ] 
                        + i[ :,    1, : ] ) * self._oo_dy
        o[ :, -1,:] = ( - i[ :,   -2, : ] 
                        + i[ :,   -1, : ] ) * self._oo_dy




    def diffZC(self, i, o ): 
        ''' 
            @summary: Calculates the central difference of an input scalar field in the z-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function
        '''
        o[:,:,1:-1] = ( - i[ :, :, 0:-2 ]   
                        + i[ :, :, 2:   ] ) * self._oo_2dz
        o[:, :,  0] = ( - i[ :, :,    0 ] 
                        + i[ :, :,    1 ] ) * self._oo_dz
        o[:, :, -1] = ( - i[ :, :,   -2 ] 
                        + i[ :, :,   -1 ] ) * self._oo_dz


        
        
    def diffXF( self, i, o ):
        ''' 
            @summary: Calculates the forward difference of an input scalar field in the x-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        # Forward difference used:
        # pos.    | x- | x0 | x+ |
        # coeff.  |  0 | -1 | +1 |
        #
        o[0:-1,:,:] = ( - i[ 0:-1, :, : ]   
                        + i[ 1:  , :, : ] ) * self._oo_dx

        # Switch to backward difference only at upper end
        o[  -1,:,:] = ( - i[ -2, :, : ] 
                        + i[ -1, :, : ] ) * self._oo_dx

    
    
    
    def diffYF( self, i, o ):
        ''' 
            @summary: Calculates the forward difference of an input scalar field in the y-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        o[:,0:-1,:] = ( - i[ :, 0:-1, : ]   
                        + i[ :, 1:  , : ] ) * self._oo_dy

        o[:,-1,:] = ( - i[ :, -2, : ] 
                      + i[ :, -1, : ] ) * self._oo_dy
               

    
    
    def diffZF( self, i, o ):
        ''' 
            @summary: Calculates the forward difference of an input scalar field in the z-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        o[:,:,0:-1] = ( - i[ :, :, 0:-1]   
                        + i[ :, :, 1:  ] ) * self._oo_dz

        o[:,:,-1] = ( - i[ :, :, -2] 
                      + i[ :, :, -1] ) * self._oo_dz





    def diffXB( self, i, o ):
        ''' 
            @summary: Calculates the backward difference of an input scalar field in the x-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        # Forward difference used:
        # pos.    | x- | x0 | x+ |
        # coeff.  | -1 | +1 |  0 |
        #
        o[1:,:,:]   = ( - i[ 0:-1, :, : ]   
                        + i[ 1:  , :, : ] ) * self._oo_dx

        # Switch to forward difference only at lower end
        o[0,:,:] = ( - i[ 0, :, : ] 
                     + i[ 1, :, : ] ) * self._oo_dx




    def diffYB( self, i, o ):
        ''' 
            @summary: Calculates the backward difference of an input scalar field in the y-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        o[:,1:,:]   = ( - i[ :, 0:-1, : ]   
                        + i[ :, 1:  , : ] ) * self._oo_dy

        # Switch to forward difference only at lower end
        o[:,0,:] = ( - i[ :, 0, : ] 
                     + i[ :, 1, : ] ) * self._oo_dy




    def diffZB( self, i, o ):
        ''' 
            @summary: Calculates the backward difference of an input scalar field in the z-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        o[:,:,1:]   = ( - i[ :, :, 0:-1 ]   
                        + i[ :, :, 1:   ] ) * self._oo_dz

        # Switch to forward difference only at lower end
        o[:,:,0] = ( - i[ :, :, 0 ] 
                     + i[ :, :, 1 ] ) * self._oo_dz




    def diffXC2( self, i, o ):
        ''' 
            @summary: Calculates the central difference of an input scalar field in the x-direction 
                      with 4th order accuracy using a wider kernel of size 5 and writes it into the 
                      output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        #
        # 4th order accuracy everywhere it is possible
        #
        #i[4:  ,:,:] * ( -1.0 /12.0 ) # -1/12 ->  -1  x+2 element
        #i[3:-1,:,:] * (  2.0 / 3.0 ) # +2/3  ->  +8  x+1 element
        #i[2:-2,:,:] * (  0.0       ) # 0     ->   0  x (central) element
        #i[1:-3,:,:] * ( -2.0 / 3.0 ) # -2/3  ->  -8  x-1 element
        #i[0:-4,:,:] * (  1.0 /12.0 ) # +1/12 ->  +1  x-2 element
        
        o[2:-2,:,:] = (  - 1.0 * i[4:  ,:,:]
                         + 8.0 * i[3:-1,:,:]
                         - 8.0 * i[1:-3,:,:]
                         + 1.0 * i[0:-4,:,:] ) * self._oo_12dx

        # simple central difference at borders...
        o[ 1,:,:] = ( - i[  0, :, : ]   
                      + i[  2, :, : ] ) * self._oo_2dx
        o[-2,:,:] = ( - i[ -3, :, : ]   
                      + i[ -1, :, : ] ) * self._oo_2dx
                      
        # and forward/backward difference everywhere else
        o[   0,:,:] = ( - i[    0, :, : ] 
                        + i[    1, :, : ] ) * self._oo_dx
        o[  -1,:,:] = ( - i[   -2, :, : ] 
                        + i[   -1, :, : ] ) * self._oo_dx
                        
                        
                        
                        
    def diffYC2( self, i, o ):
        ''' 
            @summary: Calculates the central difference of an input scalar field in the y-direction 
                      with 4th order accuracy using a wider kernel of size 5 and writes it into the 
                      output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        #
        # 4th order accuracy everywhere it is possible
        #
        o[:,2:-2,:] = (  - 1.0 * i[:,4:  ,:]
                         + 8.0 * i[:,3:-1,:]
                         - 8.0 * i[:,1:-3,:]
                         + 1.0 * i[:,0:-4,:] ) * self._oo_12dy

        # simple central difference at borders...
        o[:, 1,:] = ( - i[ :,0, :]   
                      + i[ :,2, :] ) * self._oo_2dy
        o[:,-2,:] = ( - i[ :,-3,:]   
                      + i[ :,-1,:] ) * self._oo_2dy
                      
        # and forward/backward difference everywhere else
        o[ :,  0,:] = ( - i[  :,  0, : ] 
                        + i[  :,  1, : ] ) * self._oo_dy
        o[ :, -1,:] = ( - i[  :, -2, : ] 
                        + i[  :, -1, : ] ) * self._oo_dy

    
    
    
    def diffZC2( self, i, o ):
        ''' 
            @summary: Calculates the central difference of an input scalar field in the y-direction 
                      with 4th order accuracy using a wider kernel of size 5 and writes it into the 
                      output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        #
        # 4th order accuracy everywhere it is possible
        #
        o[:,:,2:-2] = (  - 1.0 * i[:,:,4:  ]
                         + 8.0 * i[:,:,3:-1]
                         - 8.0 * i[:,:,1:-3]
                         + 1.0 * i[:,:,0:-4] ) * self._oo_12dz

        # simple central difference at borders...
        o[:,:, 1] = ( - i[ :,:,0]   
                      + i[ :,:,2] ) * self._oo_2dz
        o[:,:,-2] = ( - i[ :,:,-3]   
                      + i[ :,:,-1] ) * self._oo_2dz
                      
        # and forward/backward difference everywhere else
        o[ :, :, 0] = ( - i[ :, :,  0 ] 
                        + i[ :, :,  1 ] ) * self._oo_dz
        o[ :, :,-1] = ( - i[ :, :, -2 ] 
                        + i[ :, :, -1 ] ) * self._oo_dz



        
    def diffXX( self, i, o ):
        ''' 
            @summary: Calculates the second difference (1, -2, 1) of an input scalar field in the x-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        # central        
        o[1:-1,:,:] =  (   1.0 * i[ 0:-2, :, : ]  
                         - 2.0 * i[ 1:-1, :, : ] 
                         + 1.0 * i[ 2:  , :, : ] ) * self._oo_dx2
        
        # switch to forward backward which is equivalent to copying the data      
        o[ 0,:,:] = o[ 1,:,:]
        o[-1,:,:] = o[-2,:,:]
                        
                        
    
    
    def diffYY( self, i, o ):        
        ''' 
            @summary: Calculates the second difference (1, -2, 1) of an input scalar field in the y-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        o[:,1:-1,:] =  (   1.0 * i[ :, 0:-2, : ]  
                         - 2.0 * i[ :, 1:-1, : ] 
                         + 1.0 * i[ :, 2:  , : ] ) * self._oo_dy2
    
        # switch to forward backward which is equivalent to copying the data      
        o[ :,0,:] = o[:, 1,:]
        o[:,-1,:] = o[:,-2,:]
                        
    

    
    def diffZZ( self, i, o ):        
        ''' 
            @summary: Calculates the second difference (1, -2, 1) of an input scalar field in the z-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        o[:,:,1:-1] =  (   1.0 * i[ :, :, 0:-2 ]  
                         - 2.0 * i[ :, :, 1:-1 ] 
                         + 1.0 * i[ :, :, 2:   ] ) * self._oo_dz2

        # switch to forward backward which is equivalent to copying the data      
        o[ :,:,0] = o[:,:, 1]
        o[:,:,-1] = o[:,:,-2]
                        



    def diffXY( self, i, o ):
        ''' 
            @summary: Calculates the mixed discrete derivatives of an input scalar field in the xy-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        o[1:-1, 1:-1, : ] = (   i[ 2:  , 2:  , : ]  
                              + i[ 0:-2, 0:-2, : ]  
                              - i[ 2:  , 0:-2, : ] 
                              - i[ 0:-2, 2:  , : ] ) * self._oo_4dxdy
        
        o[0, 1:-1, : ] = (   i[ 1, 2:  , : ]  
                           + i[ 0, 0:-2, : ]  
                           - i[ 1, 0:-2, : ] 
                           - i[ 0, 2:  , : ] ) * self._oo_2dxdy
        
        o[-1, 1:-1, : ] = (   i[ -1, 2:  , : ]  
                            + i[ -2, 0:-2, : ]  
                            - i[ -1, 0:-2, : ] 
                            - i[ -2, 2:  , : ] ) * self._oo_2dxdy
        
        o[1:-1, 0, : ] = (   i[2:  , 1, : ]  
                           + i[0:-2, 0,  : ]  
                           - i[0:-2, 1, : ] 
                           - i[2:  , 0,  : ] ) * self._oo_2dxdy
        
        o[1:-1, -1, : ] = (   i[ 2:  , -1,  : ]  
                            + i[ 0:-2, -2,  : ]  
                            - i[ 0:-2, -1,  : ] 
                            - i[ 2:  , -2,  : ] ) * self._oo_2dxdy
        
        o[0, 0, : ] = (   i[ 1, 1, : ]  
                        + i[ 0, 0, : ]  
                        - i[ 1, 0, : ] 
                        - i[ 0, 1, : ] ) * self._oo_dxdy
        
        o[-1, -1, : ] = (   i[-1,-1, : ]  
                          + i[-2,-2, : ]  
                          - i[-1,-2, : ] 
                          - i[-2,-1, : ] ) * self._oo_dxdy
        
        o[0, -1, : ] = (   i[ 1,-1, : ]  
                         + i[ 0,-2, : ]  
                         - i[ 1,-2, : ] 
                         - i[ 0,-1, : ] ) * self._oo_dxdy

        o[-1, 0, : ] = (   i[ -1,1, : ]  
                         + i[ -2,0, : ]  
                         - i[ -2,1, : ] 
                         - i[ -1,0, : ] ) * self._oo_dxdy
        
        
        
        
    def diffXZ( self, i, o ):
        ''' 
            @summary: Calculates the mixed discrete derivatives of an input scalar field in the xz-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        o[1:-1, :, 1:-1] = (   i[ 2:  , :, 2:  ]  
                             + i[ 0:-2, :, 0:-2]  
                             - i[ 2:  , :, 0:-2] 
                             - i[ 0:-2, :, 2:  ] ) * self._oo_4dxdz

        o[0, :, 1:-1 ] = (   i[ 1, :,  2:  ]  
                           + i[ 0, :,  0:-2]  
                           - i[ 1, :,  0:-2] 
                           - i[ 0, :,  2:  ] ) * self._oo_2dxdz
        
        o[-1, :,  1:-1] = (   i[ -1, :,  2:  ]  
                            + i[ -2, :,  0:-2]  
                            - i[ -1, :,  0:-2] 
                            - i[ -2, :,  2:  ] ) * self._oo_2dxdz
        
        o[1:-1, :,  0] = (   i[2:  , :,  1]  
                           + i[0:-2, :,  0]  
                           - i[0:-2, :,  1] 
                           - i[2:  , :,  0] ) * self._oo_2dxdz
        
        o[1:-1, :,  -1] = (   i[ 2:  , :,  -1]  
                            + i[ 0:-2, :,  -2]  
                            - i[ 0:-2, :,  -1] 
                            - i[ 2:  , :,  -2] ) * self._oo_2dxdz
        
        o[0, :,  0] = (   i[ 1, :,  1]  
                        + i[ 0, :,  0]  
                        - i[ 1, :,  0] 
                        - i[ 0, :,  1] ) * self._oo_dxdz
        
        o[-1, :,  -1] = (   i[-1, :, -1]  
                          + i[-2, :, -2]  
                          - i[-1, :, -2] 
                          - i[-2, :, -1] ) * self._oo_dxdz
        
        o[0, :,  -1] = (   i[ 1, :, -1]  
                         + i[ 0, :, -2]  
                         - i[ 1, :, -2] 
                         - i[ 0, :, -1] ) * self._oo_dxdz

        o[-1, :, 0] = (    i[ -1, :, 1]  
                         + i[ -2, :, 0]  
                         - i[ -2, :, 1] 
                         - i[ -1, :, 0] ) * self._oo_dxdz
                         
                         
                         
                         
    def diffYZ( self, i, o ):
        ''' 
            @summary: Calculates the mixed discrete derivatives of an input scalar field in the yz-direction 
                      and writes it into the output field provided.
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        o[:, 1:-1, 1:-1] = (   i[ :, 2:  , 2:  ]  
                             + i[ :, 0:-2, 0:-2]  
                             - i[ :, 2:  , 0:-2] 
                             - i[ :, 0:-2, 2:  ] ) * self._oo_4dydz
               
        o[:, 0, 1:-1 ] = (   i[:,  1,  2:  ]  
                           + i[ :, 0,  0:-2]  
                           - i[ :, 1,  0:-2] 
                           - i[ :, 0,  2:  ] ) * self._oo_2dydz
        
        o[:, -1,  1:-1] = (   i[ :, -1,  2:  ]  
                            + i[ :, -2,  0:-2]  
                            - i[ :, -1,  0:-2] 
                            - i[ :, -2,  2:  ] ) * self._oo_2dydz
        
        o[:, 1:-1,  0] = (   i[:, 2:  ,  1]  
                           + i[:, 0:-2,  0]  
                           - i[:, 0:-2,  1] 
                           - i[:, 2:  ,  0] ) * self._oo_2dydz
        
        o[:, 1:-1,  -1] = (   i[:,  2:  ,  -1]  
                            + i[:,  0:-2,  -2]  
                            - i[:,  0:-2,  -1] 
                            - i[:,  2:  ,  -2] ) * self._oo_2dydz
        
        o[:, 0,  0] = (   i[:,  1,  1]  
                        + i[ :, 0,  0]  
                        - i[ :, 1,  0] 
                        - i[ :, 0,  1] ) * self._oo_dydz
        
        o[:, -1,  -1] = (   i[:, -1, -1]  
                          + i[:, -2, -2]  
                          - i[:, -1, -2] 
                          - i[:, -2, -1] ) * self._oo_dydz
        
        o[:, 0,  -1] = (   i[ :, 1, -1]  
                         + i[ :, 0, -2]  
                         - i[ :, 1, -2] 
                         - i[ :, 0, -1] ) * self._oo_dydz

        o[:, -1, 0] = (   i[ :, -1,  1]  
                         + i[ :, -2, 0]  
                         - i[ :, -2, 1] 
                         - i[ :, -1, 0] ) * self._oo_dydz




    def avgXF( self, i, o ):
        ''' 
            @summary: Calculates the forward average of an input scalar field in the x-direction 
                      and writes it into the output field provided. This can be interpreted that
                      the value of the average is located at inter-node positions
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''
        # Forward average used:
        # pos.    | x- | x0 | x+ |
        # coeff.  |  0 | .5 | .5 |
        #
        o[0:-1,:,:] = (   i[ 0:-1, :, : ]   
                        + i[ 1:  , :, : ] ) * 0.5

        # Switch to backward average at upper end
        o[  -1,:,:] = (   i[ -2, :, : ] 
                        + i[ -1, :, : ] ) * 0.5

    
    
    
    def avgYF( self, i, o ):
        ''' 
            @summary: Calculates the forward average of an input scalar field in the y-direction 
                      and writes it into the output field provided. This can be interpreted that
                      the value of the average is located at inter-node positions
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''

        o[0:-1,:,:] = (   i[ :, 0:-1, : ]   
                        + i[ :, 1:  , : ] ) * 0.5

        # Switch to backward average at upper end
        o[  -1,:,:] = (   i[ :, -2, : ] 
                        + i[ :, -1, : ] ) * 0.5
                        
                        
                        
                        
    def avgZF( self, i, o ):
        ''' 
            @summary: Calculates the forward average of an input scalar field in the z-direction 
                      and writes it into the output field provided. This can be interpreted that
                      the value of the average is located at inter-node positions
            @param i: Input scalar field (3D np.array)
            @param o: Output scalar field (3D np.array). Must have been allocated by calling function.
        '''

        o[0:-1,:,:] = (   i[ :, :, 0:-1 ]   
                        + i[ :, :, 1:   ] ) * 0.5

        # Switch to backward average at upper end
        o[  -1,:,:] = (   i[ :, :, -2 ] 
                        + i[ :, :, -1 ] ) * 0.5
                        
                        
   
        
        
        
