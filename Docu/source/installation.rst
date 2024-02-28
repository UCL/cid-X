Installation
============

Requirements
------------

Currently, only python version 3 is supported, and the package was developed with 3.6 and tested with version 3.9.13.
Python packages beyond the standard modules needed are listed below and also defined in the file ``requirements.txt`` in 
the base directory of the repository. Install those to your system using the package manager of your choice such as pip.

* ``nibabel==5.2.0``
* ``numpy==1.26.4``
* ``packaging==23.2``
* ``scipy==1.12.0``
* ``SimpleITK==2.3.1``

Furthermore, the following `nifty-reg <https://github.com/KCL-BMEIS/niftyreg>`_ executables need to be installed:

* ``reg_transform``
* ``reg_resample``
* ``reg_jacobian``

Either add the directory that contains those executables to the system path, or if you cannot --or do not want to-- add
these to your path, please define the path in the configuration file in section ``BATCH_POSTPROCESSING`` to the variable
``niftyRegBinDir``.


Obtain the python sources and a first dry-run
---------------------------------------------

Get the python source code of from `github <https://github.com/UCL/cid-X>`_. Clone the repository or download the zip
file.

Set up a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step is optional, but recommended. 
With the commands below a virtual environment called ``.venv`` is created and the defined requirements 
will be installed into this environment. 

* Open a command-prompt such as ``bash`` in the base directory of the repository
* Create the virtual environment by running ``python –m venv .venv``
* Activate the virtual environment: ``source .venv/bin/activate`` or equivalent
* Install the requirements into the virtual environment ``pip install –r requirements.txt`` 


Initial check
^^^^^^^^^^^^^

Make sure that you have everything in place by

* opening a command-prompt such as ``bash`` and 
* navigate to the source directory 
* call python with the *main script* of the post-processing framework which is called ``XCATdvfProcessing.py``.

You should see a basic usage message printed telling you that you might want to consider providing a configuration file.

.. code-block:: none
   :emphasize-lines: 1,2
   
   cd /path/to/sources/pyCidX
   python XCATdvfProcessing.py

   >>> Tool to post-process XCAT DVF text files.
   >>> Usage: XCATdvfProcessing.py pathToConfigFile
   >>>        - pathToConfigFile     -> Path to the configuration file that contains all parameters for processing
