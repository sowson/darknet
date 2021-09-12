This light SDK for OpenCL only installs the required component to compile an OpenCL program
It also creates the following environment variable OCL_ROOT which points to the installation path of this SDK
To include the headers and library in your project, you can then use the following
${OCL_ROOT}\include
${OCL_ROOT}\lib\x86
${OCL_ROOT}\lib\x86_64
