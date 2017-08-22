import os
import sys
import glob
import dlib
import cv2



# In this example we are going to train a pedestrian detector based on the small
# pedestrian dataset in the examples/pedestrian directory.  This means you need to supply
# the path to this pedestrian folder as a command line argument so we will know
# where it is.
if len(sys.argv) != 2:
    print(
        "Give the path to the examples directory as the argument to program"
        "execute this program by running:\n"
        "    ./pedestrian_detector.py ./examples")
    exit()
pedestrian_folder = sys.argv[1]


# Now let's do the training.  The train_simple_object_detector() function has a
# bunch of options, all of which come with reasonable default values.  The next
# few lines goes over some of these options.
options = dlib.simple_object_detector_training_options()

# Since pedestrian are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = True

# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.  You must find the best C value
# empirically by checking how well the trained detector works on a test set of
# images you haven't trained on.  Don't just leave the value set at 1.  Try a
# few different C values and see what works best for your data.
options.C = 5

# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 4

# We can tell the trainer to print it's progress to the console if we want.  
options.be_verbose = True


# training_xml_path = os.path.join(pedestrian_folder, "training.xml")
# testing_xml_path = os.path.join(pedestrian_folder, "testing.xml")

# # This function does the actual training.  It will save the final detector to
# # detector.svm.  The input is an XML file that lists the images in the training
# # dataset and also contains the positions of the pedestrian boxes.  To create your
# # own XML files you can use the imglab tool. It is a simple graphical tool for labeling objects in
# # images with boxes. 
# dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)


# # Now that we have a pedestrian detector we can test it.  The first statement tests
# # it on the training data.  It will print(the precision, recall, and then)
# # average precision.
# print("")  # Print blank line to create gap from previous output
# print("Training accuracy: {}".format(
#     dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
# # However, to get an idea if it really worked without overfitting we need to
# # run it on images it wasn't trained on.  The next line does this.  Happily, we
# # see that the object detector works perfectly on the testing images.
# print("Testing accuracy: {}".format(
#     dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))



# Using the saved detector
# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
detector = dlib.simple_object_detector("detector.svm")

# We can look at the HOG filter we learned.  It should look like a pedestrian.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)

# To See test case run 
from skimage import io
# Now let's run the detector over the images in the pedestrian folder and display the
# results.
print("Showing detections on the images on testing images")
test_image = ['../test5.jpg']
win = dlib.image_window()
for f in test_image:
    print("Processing file: {}".format(f))
    image = cv2.imread(f)
    h,w  = image.shape[:2]
    a = min(400,w)
    img = cv2.resize(image, (a,h*a/w))
    
   # img = io.imread(f)
    dets = detector(img)
    print("Number of pedestrian detected: {}".format(len(dets)))
    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()