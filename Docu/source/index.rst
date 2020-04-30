.. cid-X: Consistent invertible deformation vecorfields for the XCAT phantom documentation master file, created by
   sphinx-quickstart on Mon Feb 17 19:28:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cid-X
================

**cid-X: Consistent invertible deformation-vector-fields for the XCAT phantom**

cid-X is a python software package that makes the output XCAT deformation vector fields (DVFs)
consistent and invertible. It produces forward and backward DVFs, as well as images corresponding 
to such DVFs. 

Features
---------

Using this post-processing framework will 

* make the DVFs invertible
* preserve the sliding motion between the chest wall and the interior organs such as lung, heart and liver
* correct for potential gaps and overlaps between the sliding regions
* convert the DVFs into a file format compatible with `nifty-reg <https://github.com/KCL-BMEIS/niftyreg>`_ tools.

.. toctree::
   :maxdepth: 2
   :Caption: Contents:
   
   installation
   usage
   license_link





