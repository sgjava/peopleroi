#!/usr/bin/env python
"""
Created on Sep 9, 2013

@author: sgoldsmith

Copyright (c) Steven P. Goldsmith

All rights reserved.
"""

import ConfigParser, logging, sys, os, traceback, socket, importlib, time, datetime, numpy, cv2, cv2.cv as cv, detect.Motion, detect.People, plop.collector

class Process():
    """Main class used to acquire and process frames for people detection using motion detection ROI.
    
    sys.argv[1] = Configuration file
    sys.argv[2] = Video input file
    sys.argv[3] = Video output file
    sys.argv[4] = Mask output file
    
    ../config/test.ini ../resources/edger.avi ../resources/people-detect.avi ../resources/mask.png
    """    
    
    def __init__(self, configFileName):
        print "%s" % configFileName
        self.parser = ConfigParser.SafeConfigParser()
        # Read configuration file
        self.parser.read(configFileName)
        # Set up logger
        self.logger = logging.getLogger("Process")
        self.logger.setLevel(self.parser.get("logging", "level"))
        formatter = logging.Formatter(self.parser.get("logging", "formatter"))
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("Configuring from file: %s" % configFileName)
        self.logger.info("Logging level: %s" % self.parser.get("logging", "level"))
        self.logger.debug("Logging formatter: %s" % self.parser.get("logging", "formatter"))
        # Set video related data attributes
        self.resizeWidth = self.parser.getint("video", "resizeWidth")
        self.resizeHeight = self.parser.getint("video", "resizeHeight")
        self.showWindow = self.parser.getboolean("video", "show")
        # Set motion related data attributes
        self.kSize = eval(self.parser.get("motion", "kSize"), {}, {})
        self.alpha = self.parser.getfloat("motion", "alpha")
        self.blackThreshold = self.parser.getint("motion", "blackThreshold")
        self.maxChange = self.parser.getfloat("motion", "maxChange")
        self.startThreshold = self.parser.getfloat("motion", "startThreshold")
        self.stopThreshold = self.parser.getfloat("motion", "stopThreshold")
        self.dilateAmount = self.parser.getint("motion", "dilateAmount")
        self.erodeAmount = self.parser.getint("motion", "erodeAmount")        
        self.markObjects = self.parser.getboolean("motion", "markObjects")          
        self.boxColor = eval(self.parser.get("motion", "boxColor"), {}, {})        
        self.ignoreAreasBoxColor = eval(self.parser.get("motion", "ignoreAreasBoxColor"), {}, {})        
        self.boxThickness = self.parser.getint("motion", "boxThickness")
        if eval(self.parser.get("motion", "ignoreAreas")) == None:
                self.ignoreAreas = None
        else:
            self.ignoreAreas = numpy.array(eval(self.parser.get("motion", "ignoreAreas"), {}, {}), dtype=numpy.int32)
        self.ignoreMask = self.parser.get("motion", "ignoreMask")
        if self.ignoreMask != "":
            self.maskImg = cv2.imread(self.ignoreMask)
            self.maskImg = cv2.cvtColor(self.maskImg, cv2.COLOR_BGR2GRAY)
        else:
            self.ignoreMask = None
            self.maskImg = None
        # Set people detect related data attributes
        self.minWidth = self.parser.getint("peopleDetect", "minWidth")
        self.minHeight = self.parser.getint("peopleDetect", "minHeight")
        self.addWidth = self.parser.getint("peopleDetect", "addWidth")
        self.addHeight = self.parser.getint("peopleDetect", "addHeight")        
        self.hitThreshold = self.parser.getfloat("peopleDetect", "hitThreshold")
        self.winStride = eval(self.parser.get("peopleDetect", "winStride"), {}, {})
        self.padding = eval(self.parser.get("peopleDetect", "padding"), {}, {})
        self.scale = self.parser.getfloat("peopleDetect", "scale")
        self.finalThreshold = self.parser.getfloat("peopleDetect", "finalThreshold")
        self.useMeanshiftGrouping = self.parser.getboolean("peopleDetect", "useMeanshiftGrouping")
        self.peopleMarkObjects = self.parser.getboolean("peopleDetect", "markObjects")          
        self.peopleBoxColor = eval(self.parser.get("peopleDetect", "boxColor"), {}, {})        
        self.filteredBoxColor = eval(self.parser.get("peopleDetect", "filteredBoxColor"), {}, {})        
        self.peopleIgnoreAreasBoxColor = eval(self.parser.get("peopleDetect", "ignoreAreasBoxColor"), {}, {})        
        self.peopleBoxThickness = self.parser.getint("peopleDetect", "boxThickness")
        if eval(self.parser.get("peopleDetect", "ignoreAreas")) == None:
                self.peopleIgnoreAreas = None
        else:
            self.peopleIgnoreAreas = numpy.array(eval(self.parser.get("peopleDetect", "ignoreAreas"), {}, {}), dtype=numpy.int32)
        # Capture file
        self.logger.info("Reading video from file: %s" % sys.argv[2])
        self.capture = cv2.VideoCapture(sys.argv[2])
        self.writer = cv2.VideoWriter()
        if self.showWindow:
            cv2.namedWindow("target", cv2.CV_WINDOW_AUTOSIZE)
            cv2.moveWindow("target", 0, 0)
            cv2.namedWindow("source", cv2.CV_WINDOW_AUTOSIZE)
            cv2.namedWindow("motion", cv2.CV_WINDOW_AUTOSIZE)
            cv2.namedWindow("motion history", cv2.CV_WINDOW_AUTOSIZE)
            if self.ignoreMask != None:     
                cv2.namedWindow("mask", cv2.CV_WINDOW_AUTOSIZE)

    def showRects(self, image, rects, winX, winY):
        """Show all rectangles as ROI"""
        imgHeight, imgWidth, imgUnknown = image.shape
        curX = winX
        for x, y, w, h in rects:
            y1 = y - self.addHeight
            if y1 < 0:
                y1 = 0
            y2 = y + h + self.addHeight
            if y2 > imgHeight:
                y2 = imgHeight
            x1 = x - self.addWidth
            if x1 < 0:
                x1 = 0
            x2 = x + w + self.addWidth
            if x2 > imgWidth:
                x2 = imgWidth
            roi = image[y1:y2, x1:x2]
            winName = "%d %d %d %d" % (x, y, w, h)
            cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)
            cv2.moveWindow(winName, curX, winY)
            cv2.imshow(winName, roi)
            curX += int(w * 1.1) + 10

    def closeRects(self, rects):
        """Close all ROI windows"""
        for x, y, w, h in rects:
            cv2.destroyWindow("%d %d %d %d" % (x, y, w, h))
            
    def detectPeople(self, source, target, rects):
        """Do ROI people detection"""  
        imgHeight, imgWidth, imgUnknown = source.shape
        foundLocations = []
        for x, y, w, h in rects:
            if w > self.minWidth and h > self.minHeight:
                y1 = y - self.addHeight
                if y1 < 0:
                    y1 = 0
                y2 = y + h + self.addHeight
                if y2 > imgHeight:
                    y2 = imgHeight
                x1 = x - self.addWidth
                if x1 < 0:
                    x1 = 0
                x2 = x + w + self.addWidth
                if x2 > imgWidth:
                    x2 = imgWidth            
                self.logger.debug("detectPeople %d %d %d %d" % (y1, y2, x1, x2))
                sourceRoi = source[y1:y2, x1:x2]
                targetRoi = target[y1 * self.heightMultiplier:y2 * self.heightMultiplier, x1 * self.widthMultiplier:x2 * self.widthMultiplier]
                foundLocations = self.people.detect(sourceRoi, targetRoi)
                if len(foundLocations) > 0:
                    self.logger.debug("Detected people locations: %s" % (foundLocations))
            else:
                self.logger.debug("Width must be %d and height must be %d: w = %d, h = %d" % (self.minWidth, self.minHeight, w, h))
        return foundLocations     
    
    def run(self):
        """Video processing loop."""
        s, target = self.capture.read()
        imgHeight, imgWidth, imgUnknown = target.shape
        frames = int(self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT)) - 1
        self.logger.info("Image dimensions %dw x %dh, %d frames" % (imgWidth, imgHeight, frames))
        self.logger.info("Writing video to file: %s" % sys.argv[3])
        self.writer.open(sys.argv[3], int(self.capture.get(cv.CV_CAP_PROP_FOURCC)), 1, (imgWidth, imgHeight))
        # Create black history image
        historyImg = numpy.zeros((self.resizeHeight, self.resizeWidth), numpy.uint8)
        if self.showWindow:
            cv2.moveWindow("source", imgWidth + 66, 0)
            cv2.moveWindow("motion", imgWidth + 66, self.resizeHeight + 54)
            cv2.moveWindow("motion history", imgWidth + self.resizeWidth + 69, self.resizeHeight + 54)
            cv2.imshow("motion history", historyImg)
            if self.ignoreMask != None:     
                cv2.moveWindow("mask", imgWidth + self.resizeWidth + 69, 0)
                cv2.imshow("mask", self.maskImg)
        motion = detect.Motion.Motion(self.resizeWidth, self.resizeHeight, imgWidth, imgHeight,
                                         self.kSize, self.alpha, self.blackThreshold, self.maxChange, self.dilateAmount, self.erodeAmount,
                                         self.markObjects, self.boxColor, self.ignoreAreasBoxColor, self.boxThickness,
                                         self.ignoreAreas, self.maskImg)
        self.people = detect.People.People(self.resizeWidth, self.resizeHeight, imgWidth, imgHeight,
                                             self.hitThreshold, self.winStride, self.padding, self.scale, self.finalThreshold, self.useMeanshiftGrouping,
                                             self.peopleMarkObjects, self.peopleBoxColor, self.filteredBoxColor, self.peopleIgnoreAreasBoxColor,
                                             self.peopleBoxThickness, self.peopleIgnoreAreas)
        # Used for full size image marking
        self.widthMultiplier = imgWidth / self.resizeWidth
        self.heightMultiplier = imgHeight / self.resizeHeight
        imagesDetected = 0
        sleep = False
        start = time.time()
        for f in xrange(frames):
            source = cv2.resize(target, (self.resizeWidth, self.resizeHeight), interpolation=cv2.INTER_NEAREST)            
            movementLocations = motion.detect(source, target)
            if motion.motionPercent > self.startThreshold and len(movementLocations) > 0:
                self.logger.debug("%3.2f%% motion detected on frame %d, locations: %s" % (motion.motionPercent, f, movementLocations))
                if self.showWindow:
                    historyImg = numpy.bitwise_or(motion.grayImg, historyImg)        
                foundLocations = self.detectPeople(source, target, movementLocations)
                if len(foundLocations) > 0:
                    self.logger.debug("People detected on frame %d, locations: %s" % (f, foundLocations))
                    self.writer.write(target)
                    imagesDetected += 1
                    sleep = True
                else:
                    sleep = False                
                sleep = True
            else:
                sleep = False
            if self.showWindow:  
                cv2.imshow("target", target)
                cv2.imshow("source", source)
                cv2.imshow("motion history", historyImg)
                if motion.grayImg != None:
                    cv2.imshow("motion", motion.grayImg)
            s, target = self.capture.read()
            if self.showWindow:
                if sleep == True:
                    self.showRects (source, movementLocations, imgWidth + 66, self.resizeHeight * 2 + 84)
                    key = cv2.waitKey(-1)
                    self.closeRects (movementLocations)
                else:
                    key = cv2.waitKey(1)
                if key == 27:
                    break
        if self.showWindow:
            self.logger.info("Writing mask to file: %s" % sys.argv[4])
            # Invert image for mask (black masks motion, white detects motion)
            return_val, historyImg = cv2.threshold(historyImg, 127, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(sys.argv[4], historyImg)
        elapse = time.time() - start
        self.logger.info("%d frames, %d frames with people, elapse time: %4.2f seconds, %4.1f FPS" % (frames, imagesDetected, elapse, frames / elapse))

    def cleanUp(self):
        if self.showWindow:
            cv2.destroyWindow("target")
            cv2.destroyWindow("source")
            cv2.destroyWindow("motion")
            cv2.destroyWindow("motion history")
            if self.ignoreMask != None:     
                cv2.destroyWindow("mask")
        del self.writer

if __name__ == "__main__":
    try:
        process = Process(sys.argv[1])
        process.run()
    except:
        sys.stderr.write("%s " % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f"))
        traceback.print_exc(file=sys.stderr)
    # Do cleanup
    process.cleanUp()

