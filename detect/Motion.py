"""
Created on Mar 18, 2013

@author: sgoldsmith

Copyright (c) Steven P. Goldsmith

All rights reserved.
"""

import cv2, numpy, logging, DetectBase

class Motion(DetectBase.DetectBase):
    """Motion detector.
    
    Uses moving average to determine change percent. Object marking can be used to dial in settings
    for various conditions.
    
    """
    
    def __init__(self, sourceWidth, sourceHeight, targetWidth, targetHeight,
                 kSize, alpha, blackThreshold, maxChange, dilateAmount, erodeAmount,
                 markObjects, boxColor, ignoreAreasBoxColor, boxThickness, ignoreAreas, ignoreMask):
        # Get logger
        self.logger = logging.getLogger("VideoLoop")
        # Set class attributes  
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight
        self.kSize = kSize
        self.alpha = alpha
        self.blackThreshold = blackThreshold
        self.maxChange = maxChange
        self.dilateAmount = dilateAmount
        self.erodeAmount = erodeAmount
        self.markObjects = markObjects
        self.boxColor = boxColor
        self.ignoreAreasBoxColor = ignoreAreasBoxColor
        self.boxThickness = boxThickness
        self.ignoreAreas = ignoreAreas
        self.ignoreMask = ignoreMask
        # Used for full size image marking
        self.widthMultiplier = targetWidth / sourceWidth
        self.heightMultiplier = targetHeight / sourceHeight
        # Set the rest of the data attributes
        self.totalPixels = self.sourceWidth * self.sourceHeight
        self.grayImg = None
        self.movingAvgImg = None
        self.motionPercent = 0.0

    def contours(self, source):
        # The background (bright) dilates around the black regions of frame
        source = cv2.dilate(source, None, iterations=self.dilateAmount);
        # The bright areas of the image (the background, apparently), get thinner, whereas the dark zones bigger
        source = cv2.erode(source, None, iterations=self.erodeAmount);
        # Find contours
        image, contours, heirarchy = cv2.findContours(source, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Add objects with motion
        movementLocations = []
        for contour in contours:
            rect = cv2.boundingRect(contour)
            # See if we should ignore any areas
            if self.ignoreAreas == None:
                movementLocations.append(rect)
            elif not self.insideIgnoreAreas(rect):
                movementLocations.append(rect)
        return movementLocations        
    
    def detect(self, source, target):
        """Motion detection using OpenCV.
        
        Based on moving average image.
        
        """
        
        movementLocations = []
        # Generate work image by blurring.
        self.workImg = cv2.blur(source, self.kSize)
        # Generate moving average image if needed
        if self.movingAvgImg == None:
            self.movingAvgImg = numpy.float32(self.workImg)
        # Generate moving average image
        cv2.accumulateWeighted(self.workImg, self.movingAvgImg, self.alpha)
        self.diffImg = cv2.absdiff(self.workImg, cv2.convertScaleAbs(self.movingAvgImg))
        # Convert to grayscale
        self.grayImg = cv2.cvtColor(self.diffImg, cv2.COLOR_BGR2GRAY)
        # Convert to BW
        return_val, self.grayImg = cv2.threshold(self.grayImg, self.blackThreshold, 255, cv2.THRESH_BINARY)
        # Apply ignore mask
        if self.ignoreMask != None:
            self.grayImg = numpy.bitwise_and(self.grayImg, self.ignoreMask)        
        # Total number of changed motion pixels
        self.motionPercent = 100.0 * cv2.countNonZero(self.grayImg) / self.totalPixels
        # Detect if camera is adjusting and reset reference if more than maxChange
        if self.motionPercent > self.maxChange:
            self.logger.debug("%3.1f%% motion detected, resetting reference image" % self.motionPercent)                    
            self.movingAvgImg = numpy.float32(self.workImg)
        movementLocations = self.contours(self.grayImg)
        # Mark objects (make sure to copy target image if you want to keep original image intact)
        if self.markObjects == True:
            self.mark(source, target, movementLocations, self.widthMultiplier, self.heightMultiplier, self.boxColor)
            if self.ignoreAreas != None:                
                self.mark(source, target, self.ignoreAreas, self.widthMultiplier, self.heightMultiplier, self.ignoreAreasBoxColor)
        # Return filtered results
        return movementLocations
    
