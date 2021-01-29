Installation
============

Requirements
------------

Currently only python version 3 is supported and the package was developed and tested with version 3.6.
Python packages needed are listed below. Install those to your system using the package manager of your 
choice such as pip or anaconda.

* ``configparser``
* ``nibabel``
* ``numpy``
* ``scipy``

Furthermore the following `nifty-reg <https://github.com/KCL-BMEIS/niftyreg>`_ executables need to be installed:

* ``reg_transform``
* ``reg_resample``
* ``reg_jacobian``

Either add the directory that contains those executables to the system path, or if you cannot --or do not want to--  add
these to your path, please define the path in the configuration file in section ``BATCH_POSTPROCESSING`` to the variable
``niftyRegBinDir``.

Obtain the python sources and a first dry-run
---------------------------------------------

Get the python source code of from `github <https://github.com/UCL/cid-X>`_. Clone the repository or download the zip
file.

Try that you have everything in place by

* opening a command-prompt such as ``bash`` and 
* navigate to the source directory 
* call python with the *main script* of the post-processing framework which is called ``XCATdvfProcessing.py``.

You sould see a basic usage message printed telling you that you might want to consider providing a configuration file.

.. code-block:: none
   :emphasize-lines: 1,2
   
   cd /path/to/sources/pyCidX
   python XCATdvfProcessing.py

   >>> Tool to post-process XCAT DVF text files.
   >>> Usage: XCATdvfProcessing.py pathToConfigFile
   >>>        - pathToConfigFile     -> Path to the configuration file that contains all parameters for processing
