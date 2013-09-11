"""
Created on Mar 18, 2013

@author: sgoldsmith

Copyright (c) Steven P. Goldsmith

All rights reserved.
"""

import abc, cv2

class DetectBase():
    """Detect abstract base class.
    
    Override detect function to do any image processing.

    """

    __metaclass__ = abc.ABCMeta

    def inside(self, r, q):
        """See if one rectangle inside another"""
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh
    
    def insideIgnoreAreas(self, r):
        """See if rectangle inside areas that are ignored"""
        inside = False
        for area in self.ignoreAreas:
            # If at least inside one ignore area
            if self.inside(r, area):
                inside = True
                break
        return inside
    
    def mark(self, source, target, rects, widthMul, heightMul, boxColor):
        """Mark detected objects in image"""
        for x, y, w, h in rects:
            # Mark source
            cv2.rectangle(source, (x, y), (x + w, y + h), boxColor, self.boxThickness)
            # Mark target
            cv2.rectangle(target, (x * widthMul, y * heightMul),
                          ((x + w) * widthMul, (y + h) * heightMul),
                          boxColor, self.boxThickness)
    
    @abc.abstractmethod
    def detect(self, source, target):
        """Return numpy array of areas detected.
        
        source is usually a down sized image for performance. target is the 
        original image which may be modified by this method.
        
        """
        raise NotImplementedError, "Please override in the derived class"

