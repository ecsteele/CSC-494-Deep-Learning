# NkuMyaDevMaker.py
# NKU CSC/DSC 494/594 Spring 2018 Deep Learning (K Kirby)
# -----------------------------------------------------------------
#
# For generating some training sets for image classification,
# for simple experimentation with machine learning.
#
# Images consist of scaled, translated and rotated versions of
# Myanmar characters (as used in the Burmese language) and 
# Devanagari characters (as used in the Hindi language).
#
# Test and training sets use different characters. The idea
# is to see whether a neural network can learn to distinguish
# these character sets because of some inherent geometric features.
#
# This example was developed in collaboration with NKU students
# Lauren Hensley and Brian Konzman in 2016.
#
# Needs the two font files referenced below in the working directory.
#
# Run demo_showExampleImages() below to see some examples of these
# generated images.
#
# Returns data sets as pairs (X,Y) so you can use this with
# scikit-learn or TensorFlow classifiers as well as our own 
# home-grown classifiers.
# 
# Each input pattern in X is a flat numpy array of float32 
# numbers between 0.0 and 1.0 indicating graylevels of
# a flattened square image of given size.
#
# Each output pattern in Y is a single float32: 0.0 (Devanagari)
# or 1.0 (Myanmar).
#
# See:
# https://en.wikipedia.org/wiki/Devanagari_(Unicode_block)
# https://en.wikipedia.org/wiki/Myanmar_(Unicode_block)
# -----------------------------------------------------------------


# The usual imports
import random
import numpy as np
import matplotlib.pyplot as plt

# Python Image Library (old but unlike OpenCV installed with Anaconda)
from PIL import Image, ImageFont, ImageDraw

# You will need these font files in  the directory this program runs in.
MYANMAR_FONT    = 'Myanmar MN.ttc'
DEVANAGARI_FONT = 'DevanagariMT.ttc' 

# Pick some codepoints
MYANMAR_CODEPOINTS_TRAIN = list(range(0x1010,0x1022))  # 18 Myanmar chars
MYANMAR_CODEPOINTS_TEST  = list(range(0x1000,0x1010))  # 16 Myanmar chars

DEVANAGARI_CODEPOINTS_TRAIN = list(range(0x0920,0x0932)) # 18 Devanagari chars
DEVANAGARI_CODEPOINTS_TEST  = list(range(0x0910,0x0920)) # 16 Devanagari chars


def makeDataSet(n, numExamples, training ):
    """Data set: x's are flattened nxn images of char, y's indicate 0/1 (devanagari/myanmar)."""
    
    # The boolean parameter training indicates whether we use train or test code points.
    MYANMAR = MYANMAR_CODEPOINTS_TRAIN if training else MYANMAR_CODEPOINTS_TEST
    DEVANAGARI = DEVANAGARI_CODEPOINTS_TRAIN if training else DEVANAGARI_CODEPOINTS_TEST

    # Generate random (x,y) pairs and append them to the training set.
    X = []
    Y = []
    for i in range(numExamples):
        myanmar = random.randrange(2)  # flip a coin: myanmar or devanagari?
        if myanmar:
            img = makeCharImage(n, MYANMAR_FONT, random.choice(MYANMAR))  
        else:
            img = makeCharImage(n, DEVANAGARI_FONT, random.choice(DEVANAGARI))  
        X.append(img.ravel())    # flatten to a 1D numpy array
        Y.append(float(myanmar)) # the correct classification is this
    return (X,Y)



def makeCharImage( imageSize, fontName, codePoint ):
    """ Generate a randomly rotate/scaled image of a character in this font."""
    
    # Set foreground and background colors (graylevels). 
    # Note there is antialiasing, so some pixel values will be between 0 and 255.
    bgGraylevel =    0   
    fgGraylevel =  255   
    
    # Get the unicode character at this codepoint. 
    char= chr(codePoint)
    
    # Construct empty graylevel image.
    img = Image.new('L', (imageSize, imageSize), bgGraylevel) # 'L' means graylevel (not RGB)
    
    # Get randomly sized font and its dimensions in pixels.
    minSize = int(0.60*imageSize)
    maxSize = int(0.85*imageSize)
    fontSize =  random.randint(minSize, maxSize)
    font = ImageFont.truetype(fontName, fontSize)
    (font_width, font_height) = font.getsize(char)
    
    # Construct a tiny image just big enough to hold any rotated letter.  
    imgLetter = Image.new('L', (font_width, font_height), bgGraylevel)
    ImageDraw.Draw(imgLetter).text((0, 0), char, fgGraylevel, font=font)
    
    # Do the same thing but construct a mask.
    mask = Image.new('L', (font_width, font_height), 0)
    ImageDraw.Draw(mask).text((0, 0), char, 255, font=font)
    
    # Make a rotated copy of each.
    angle = random.randint(-179,180)
    imgLetterRotated = imgLetter.rotate(angle, expand=True)
    maskRotated = mask.rotate(angle,expand=True)
    width, height = imgLetterRotated.size
    
    # Paste rotated letter on into background image.
    x = random.randint(0, max(0, imageSize - width - 1))
    y = random.randint(0, max(0, imageSize - height - 1))
    img.paste(imgLetterRotated, (x, y), maskRotated)
    
    # Return image as 2D numpy array scaled to between 0.0 and 1.0.
    return np.array(img,dtype=np.float32)/255
 


def demo_showExampleImages():  
    """Displays some example images generated for use in data sets."""
    
    N= 128 # We will generate NxN images (as 2D numpy arrays)   

    # Show 10 examples from a training set.     
    X,Y = makeDataSet( N, 10, training=True )
    for i in range(len(X)):
        img= np.array(X[i]).reshape(N,N)
        plt.imshow(img)
        plt.show()
        print(Y[i])

        
#demo_showExampleImages()
