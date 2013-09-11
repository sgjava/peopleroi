# People detection using motion ROI

This compares and contrasts using image resizing and motion detection to obtain
a ROI (region of interest) versus processing frames and regions unchanged.
People detection is used in this example, but any type of image analysis can
benefit from the techniques contained within.

### System requirements

All development and testing were performed on Ubuntu 64 bit Server/Desktop version
12.04.3 x86_64, 12.10 x86_64 and 32 bit 12.10 armv7l using PicUntu. The code and
dependencies will surely work on other Linux distributions and perhaps Windows.

### Installation

1. Install OpenCV (choose which version)
    * OpenCV 2.4.6.1 from source (best choice):
        * `sudo su -`
        * `wget https://raw.github.com/sgjava/cvp/master/scripts/ubuntu1204/install-opencv2461.sh`
        * `chmod a+x install-opencv2461.sh`
        * `./install-opencv2461.sh` (this will take a while, go smoke a cigar. On RK3066 based Mini PC it can take 4 or 5 hours.)
    * OpenCV 2.3.1 pre-built packages (I'm not supporting this version, so you will have to fix up the code as needed):
        * `sudo apt-get install python-opencv`
2. Install the rest of the dependencies:
    * `sudo su -`
    * `easy_install plop`
    * `easy_install tornado`

### FreeBSD License

Copyright (c) Steven P. Goldsmith

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
