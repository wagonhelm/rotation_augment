Put images and labels with matching filename into a folder labeled `data` in git repositories root directory.<br>  This script will translate the `yolo` bounding box data structure ([center x, center y, width height]) into `tf` bounding box data structure ([y_min, x_min, y_max, x_max]) in order to do rotations, then will translate back into `yolo` format.

1. If you are uncertain if your data is good you can choose option 1 and look through data.  If you choose to seperate sorted data when you press spacebar it will move bad data into the `badData` folder and will move good data into a folder called `goodData` when you press any other key.  Press ESC to escape.

2. Option 2 will create augmented rotated data and save in the `augImages` directory.  It will only rotate the image if the bounding boxes edges are not within 5% of the images edges.  This prevents losing bounding boxes / skewing data.

3. Option 3 will look through augmented data with bounding boxes drawn. If you choose to seperate sorted data it will move bad data into a folder called `badData` when you press the spacebar and will move good data into a folder called `goodData` when you press any other key.  Press ESC to escape.

4. If you are happy with augmented data choose option 4 to move it into your `data` folder.

5. Option 5 will create a train text file for ARVP's darknet.

Augment away!
