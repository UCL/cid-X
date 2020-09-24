#!/usr/bin/env python3


import findExecutable
import shlex, time, subprocess, os


def runCommand( cmdIn, paramsIn, logFileName=None, workDir=None, onlyPrintCommand=False ) :
    
    if workDir != None :
        curDir = os.getcwd()
        os.chdir( workDir )
        
    # Is the executable available?
    if findExecutable.findExecutable( cmdIn ) == None:
        print('Cannot find %s in path. Sorry.' % cmdIn )
        return
    
    print('Executing:\n -> %s %s' % ( cmdIn, paramsIn ) )
    cmd = shlex.split ( cmdIn + ' ' + paramsIn, posix=False ) 

    tic = time.perf_counter()
    ret = 0
    
    if not onlyPrintCommand :
        if logFileName == None :
            ret=subprocess.Popen( cmd ).wait()
            
        else:
            with open( logFileName, 'a' ) as logFile:
                logFile.write( 'Executing:\n -> %s %s\n\n' % ( cmdIn, paramsIn ) )
                logFile.flush()
                ret = subprocess.Popen( cmd, stdout=logFile, stderr=logFile ).wait()
    
        print('Return code: ' + str(ret) )
        
    if workDir != None :
        os.chdir( curDir )
    
    toc = time.perf_counter()
    
    print('Done. This took %.2fs' %( toc-tic ) )
    return ret




if __name__ == '__main__' :
    
    cmd = 'niftkAdd'
    params = '-h -d "D:\\asdf\\asdf\\asdf 64Bit"'
    fileOut = 'testLog.txt'
    runCommand( cmd, params, fileOut, onlyPrintCommand=False )
    pass
