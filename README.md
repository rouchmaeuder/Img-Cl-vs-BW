# tiff file parser
>[!warning]
>this is a incomplete project under developement. use at own risk.

code in master is mostly working on my system:
- Win 11
- WSL2
- Cuda toolkit for WSL

a tif file is necessary that is in
- little endian
- in wich the image data is stored in a singular data strip
- in backtoback data compression
> [!important]
> this does not comply with full tiff specification and is only designed to be able to analyze the output of Photoshop exporting a so called uncompressed tiff with fast en/decoding but larger size.
