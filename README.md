# perikymata-CPSC448
Program to identify and measure perikymata on high resolution images of teeth

## Setup
Clone the repository and run the following commands in the idPerikymata directory:

```
cmake .
make
```

## Use
The The program takes 2 arguments: the path of the image and the location (in pixels) of the transect to take down the image. Run the program like this:

```
./idPerikymata exampleImage.jpg 25
```

The program will write 4 files to disk: a processed image with the perikymata marked with stars along the chosen transect, a histogram of the pixel intensity values along the transect, a csv of the coordinates of the perikymata, and a csv of the distances between them.
