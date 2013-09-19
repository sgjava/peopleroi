#!/usr/bin/env python
"""
Created on Sep 9, 2013

@author: sgoldsmith

Copyright (c) Steven P. Goldsmith

All rights reserved.
"""

import ConfigParser, logging, sys, os, traceback, time, datetime, numpy, cv2, cv2.cv as cv, detect.Motion, detect.People, cProfile, pstats

class Process():
    """Main class used to acquire and process frames for people detection using motion detection ROI.
    
    Three methods are compared:
    
    1. Full size image
    2. Resized image
    3. Resized image using motion detection to identify ROIs
    
    sys.argv[1] = Configuration file
    sys.argv[2] = Video input file
    sys.argv[3] = Video output path
    sys.argv[4] = Mask output file
    
    ../config/test.ini ../resources/edger.avi ../output/ ../output/mask.png
    
    """    
    
    def __init__(self, configFileName):
        """Read in configuration, set up video writer and setup windows"""
        self.parser = ConfigParser.SafeConfigParser()
        # Read configuration file
        self.parser.read(configFileName)
        # Set up logger
        self.logger = logging.getLogger("Process")
        self.logger.setLevel(self.parser.get("logging", "level"))
        # Only add handler once
        if self.logger.handlers == []:
            formatter = logging.Formatter(self.parser.get("logging", "formatter"))
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.info("Configuring from file: %s" % configFileName)
        self.logger.info("Logging level: %s" % self.parser.get("logging", "level"))
        self.logger.debug("Logging formatter: %s" % self.parser.get("logging", "formatter"))
        self.profile = self.parser.getboolean("profiling", "profile")
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
        # Setup windows
        if self.showWindow:
            cv2.namedWindow("target", cv2.CV_WINDOW_AUTOSIZE)
            cv2.moveWindow("target", 0, 0)
            cv2.namedWindow("source", cv2.CV_WINDOW_AUTOSIZE)
            cv2.namedWindow("motion", cv2.CV_WINDOW_AUTOSIZE)
            cv2.namedWindow("motion history", cv2.CV_WINDOW_AUTOSIZE)
            cv2.namedWindow("motion ROI", cv2.CV_WINDOW_AUTOSIZE)
            if self.ignoreMask != None:     
                cv2.namedWindow("mask", cv2.CV_WINDOW_AUTOSIZE)

    def padRects(self, image, rects, useFilter):
        """Pad rectangles, get image dimensions, ROI composite image size for display and total ROI pixels"""
        imgHeight, imgWidth, imgUnknown = image.shape
        winWidth = 0
        winHeight = 0
        roiPixels = 0
        paddedRects = []
        # Get consolidated image width and height from rects
        for x, y, w, h in rects:
            # Filter based on size if True
            if (not useFilter) or (useFilter and w > self.minWidth and h > self.minHeight):
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
                paddedRects.append([x1, y1, x2 - x1, y2 - y1])
                winWidth += x2 - x1
                winHeight = max(winHeight, y2 - y1)
            else:
                self.logger.debug("Width must be %d and height must be %d: w = %d, h = %d" % (self.minWidth, self.minHeight, w, h))
        return paddedRects, imgHeight, imgWidth, winHeight, winWidth, roiPixels
                
    def showRects(self, image, rects):
        """Show all rectangles in a single window"""
        paddedRects, imgHeight, imgWidth, winHeight, winWidth, roiPixels = self.padRects(image, rects, False)
        # Black image
        rectsImg = numpy.zeros((winHeight, winWidth, 3), numpy.uint8)
        curX = 0
        # Render ROIs in one image
        for x, y, w, h in paddedRects:
            # Get ROI
            roi = image[y:y + h, x:x + w]
            # Add to image
            rectsImg[:roi.shape[0], curX:curX + roi.shape[1]] = roi
            curX += w
        cv2.imshow("motion ROI", rectsImg)
            
    def detectPeopleRoi(self, source, target, rects):
        """Do people detection on ROIs"""  
        paddedRects, imgHeight, imgWidth, winHeight, winWidth, roiPixels = self.padRects(source, rects, True)
        sourcePixels = imgHeight * imgWidth
        foundLocations = []
        for x, y, w, h in paddedRects:
            self.logger.debug("detectPeopleRoi %d %d %d %d" % (x, y, w, h))
            sourceRoi = source[y:y + h, x:x + w]
            targetRoi = target[y * self.heightMultiplier:y + h * self.heightMultiplier, x * self.widthMultiplier:x + w * self.widthMultiplier]
            foundLocations = self.people.detect(sourceRoi, targetRoi)
            if len(foundLocations) > 0:
                self.logger.debug("Detected people locations: %s" % (foundLocations))
        return foundLocations     
    
    def detectPeople(self, source, target):
        """Do people detection on full image"""  
        imgHeight, imgWidth, imgUnknown = source.shape
        foundLocations = self.people.detect(source, target)
        if len(foundLocations) > 0:
            self.logger.debug("Detected people locations: %s" % (foundLocations))
        return foundLocations     
  
    def run(self, useResize, useRoi):
        """Video processing loop."""
        self.logger.info("*** Resize = %s, ROI = %s ***" % (useResize, useRoi))
        s, target = self.capture.read()
        imgHeight, imgWidth, imgUnknown = target.shape
        frames = int(self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT)) - 1
        self.logger.info("Image dimensions %dw x %dh, %d frames" % (imgWidth, imgHeight, frames))
        # Build video file name
        if useResize:
            fileName = "resize-"
        else:
            fileName = "noresize-"
        if useRoi:
            fileName += "roi-"
        else:
            fileName += "noroi-"
        fileName = sys.argv[3] + fileName + os.path.basename(sys.argv[2])
        self.logger.info("Writing video to file: %s" % fileName)
        self.writer.open(fileName, int(self.capture.get(cv.CV_CAP_PROP_FOURCC)), 1, (imgWidth, imgHeight))
        # Create black history image
        historyImg = numpy.zeros((self.resizeHeight, self.resizeWidth), numpy.uint8)
        if self.showWindow:
            cv2.moveWindow("source", imgWidth + 69, 0)
            cv2.moveWindow("motion", imgWidth + 69, self.resizeHeight + 103)
            cv2.moveWindow("motion history", imgWidth + self.resizeWidth + 116, self.resizeHeight + 103)
            cv2.imshow("motion history", historyImg)
            if self.ignoreMask != None:     
                cv2.moveWindow("mask", imgWidth + self.resizeWidth + 69, 0)
                cv2.imshow("mask", self.maskImg)
            cv2.moveWindow("motion ROI", imgWidth + 69, self.resizeHeight * 2 + 183)
        motion = detect.Motion.Motion(self.resizeWidth, self.resizeHeight, imgWidth, imgHeight,
                                         self.kSize, self.alpha, self.blackThreshold, self.maxChange, self.dilateAmount, self.erodeAmount,
                                         self.markObjects, self.boxColor, self.ignoreAreasBoxColor, self.boxThickness,
                                         self.ignoreAreas, self.maskImg)
        if useResize:
            peopleWidth = self.resizeWidth
            peopleHeight = self.resizeHeight
        else:
            peopleWidth = imgWidth
            peopleHeight = imgHeight
        self.people = detect.People.People(peopleWidth, peopleHeight, imgWidth, imgHeight,
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
            if len(movementLocations) > 0:
                self.logger.debug("%3.2f%% motion detected on frame %d, locations: %s" % (motion.motionPercent, f, movementLocations))
                if self.showWindow:
                    historyImg = numpy.bitwise_or(motion.grayImg, historyImg)
                if useResize:
                    if useRoi:
                        foundLocations = self.detectPeopleRoi(source, target, movementLocations)
                    else:
                        foundLocations = self.detectPeople(source, target)
                else:
                    foundLocations = self.detectPeople(target, target)
                if self.showWindow:
                    self.showRects (source, movementLocations)
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
                if sleep == False:
                    key = cv2.waitKey(1)
                else:
                    key = cv2.waitKey(-1)
                if key == 27:
                    break
            s, target = self.capture.read()
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
            cv2.destroyWindow("motion ROI")
            if self.ignoreMask != None:     
                cv2.destroyWindow("mask")
        del self.writer
        
if __name__ == "__main__":
    try:
        process = Process(sys.argv[1])
        if process.profile:
            process.logger.info("Profiling enabled")
            cProfile.run("process.run(useResize=False, useRoi=False)", "%srestats" % sys.argv[3])
            stats = pstats.Stats("%srestats" % sys.argv[3])
            stats.strip_dirs().sort_stats('time').print_stats(10)
        else:
            process.run(useResize=False, useRoi=False)
        process.cleanUp()
        process = Process(sys.argv[1])
        if process.profile:
            process.logger.info("Profiling enabled")
            cProfile.run("process.run(useResize=True, useRoi=False)", "%srestats" % sys.argv[3])
            stats = pstats.Stats("%srestats" % sys.argv[3])
            stats.strip_dirs().sort_stats('time').print_stats(10)
        else:
            process.run(useResize=True, useRoi=False)
        process.cleanUp()
        process = Process(sys.argv[1])
        if process.profile:
            process.logger.info("Profiling enabled")
            cProfile.run("process.run(useResize=True, useRoi=True)", "%srestats" % sys.argv[3])
            stats = pstats.Stats("%srestats" % sys.argv[3])
            stats.strip_dirs().sort_stats('time').print_stats(10)
        else:
            process.run(useResize=True, useRoi=True)
        process.cleanUp()
    except:
        sys.stderr.write("%s " % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f"))
        traceback.print_exc(file=sys.stderr)
