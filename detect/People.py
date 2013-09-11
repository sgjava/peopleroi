"""
Created on Mar 5, 2013

@author: sgoldsmith

Copyright (c) Steven P. Goldsmith

All rights reserved.
"""

import cv2, DetectBase

class People(DetectBase.DetectBase):
    """Histogram of Oriented Gradients object detector.
    
    Using OpenCV cv2.HOGDescriptor().detectMultiScale()
    
    """
    
    def __init__(self, sourceWidth, sourceHeight, targetWidth, targetHeight,
                 hitThreshold, winStride, padding, scale, finalThreshold, useMeanshiftGrouping,
                 markObjects, boxColor, filteredBoxColor, ignoreAreasBoxColor, boxThickness, ignoreAreas):
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight        
        self.hitThreshold = hitThreshold
        self.winStride = winStride
        self.padding = padding
        self.scale = scale
        self.finalThreshold = finalThreshold
        self.useMeanshiftGrouping = useMeanshiftGrouping
        self.markObjects = markObjects
        self.boxColor = boxColor
        self.filteredBoxColor = filteredBoxColor
        self.ignoreAreasBoxColor = ignoreAreasBoxColor
        self.boxThickness = boxThickness
        self.ignoreAreas = ignoreAreas
        # Used for full size image marking
        self.widthMultiplier = targetWidth / sourceWidth
        self.heightMultiplier = targetHeight / sourceHeight       
        # Create HOG detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect(self, source, target):
        """People detection using OpenCV.
        
        Based on HOGDescriptor_getDefaultPeopleDetector. To get a higher hit-rate
        (and more false alarms, respectively), decrease the hitThreshold and
        groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        
        """     
      
        foundLocations, foundWeights = self.hog.detectMultiScale(source, hitThreshold=self.hitThreshold,
                                                                 winStride=self.winStride, padding=self.padding,
                                                                 scale=self.scale, finalThreshold=self.finalThreshold,
                                                                 useMeanshiftGrouping=self.useMeanshiftGrouping)
        foundLocationsFiltered = []
        # At least one person detected?
        if len(foundLocations) > 0:
            # Filter out inside rectangles
            for ri, r in enumerate(foundLocations):
                for qi, q in enumerate(foundLocations):
                    if ri != qi and self.inside(r, q):
                        break
                else:
                    # See if we should ignore any areas
                    if self.ignoreAreas == None:
                        foundLocationsFiltered.append(r)
                    elif not self.insideIgnoreAreas(r):
                        foundLocationsFiltered.append(r)
            # Mark objects (make sure to copy target image if you want to keep original image intact)
            if self.markObjects == True:
                self.mark(source, target, foundLocations, self.widthMultiplier, self.heightMultiplier, self.boxColor)
                self.mark(source, target, foundLocationsFiltered, self.widthMultiplier, self.heightMultiplier, self.filteredBoxColor)
                if self.ignoreAreas != None:                
                    self.mark(source, target, self.ignoreAreas, self.widthMultiplier, self.heightMultiplier, self.ignoreAreasBoxColor)
        # Return filtered results
        return foundLocationsFiltered
