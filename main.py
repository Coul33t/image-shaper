import random as rn
import numpy as np
import cv2

from copy import deepcopy

from PIL import Image, ImageDraw
from skimage import color

from math import sqrt

class Shape:
    def __init__(self, name: str, colour: tuple, parameters: dict):
        self.name = name
        self.colour = colour
        self.parameters = parameters

class ImageAndShapes:
    def __init__(self, img: Image, shapes: list):
        self.img = img
        self.shapes = shapes

class ImageShaper:
    def __init__(self, path_to_initial_img):
        self.shape = 'Triangle'
        # Each shape is a gene
        self.initial_img = Image.open(path).convert('RGB')
        self.initial_img_lab = cv2.cvtColor(np.asarray(self.initial_img, dtype='float32') / 255, cv2.COLOR_RGB2Lab)
        self.rgb_array_initial = np.asarray(self.initial_img)
        self.lab_array_initial = np.asarray(self.initial_img_lab)
        self.shaped_img = self.initial_img.copy().convert('RGBA')

        self.width = self.rgb_array_initial.shape[1]
        self.height = self.rgb_array_initial.shape[0]

        # Min size for the shapes (idk how to use this but I'll figure it out)
        self.min_size = (int(self.rgb_array_initial.shape[0] / 100),
                         int(self.rgb_array_initial.shape[1] / 100))

        self.mean_colour_original = self.get_mean_colour(self.initial_img)

    def get_mean_colour(self, img: Image.Image) -> np.ndarray:
        """
            Compute the mean colour from an image.
        """
        mean_pixel_value = np.asarray([0, 0, 0], np.float64)

        all_colours = img.getcolors(img.size[0] * img.size[1])

        for colour in all_colours:
            mean_pixel_value += np.asarray(colour[1]) * colour[0]

        mean_pixel_value = mean_pixel_value / (self.width * self.height)

        return mean_pixel_value.astype('uint8')

    def generate_shape(self, shape_name='Triangle'):
        new_colour = (rn.randint(0, 255),
                      rn.randint(0, 255),
                      rn.randint(0, 255))

        if shape_name == 'Triangle':
            h = self.rgb_array_initial.shape[0]
            w = self.rgb_array_initial.shape[1]
            coordinates = [rn.randint(0, w), rn.randint(0, h),
                           rn.randint(0, w), rn.randint(0, h),
                           rn.randint(0, w), rn.randint(0, h)]

        shape = Shape(shape_name, new_colour, {'coordinates': coordinates})

        return shape

    def put_shapes_in_image(self, base_image_and_shape):
        initial_transp = base_image_and_shape.img.convert('RGBA')

        for shape in base_image_and_shape.shapes:
            transparent_img = Image.new('RGBA', self.initial_img.size, (255, 255, 255, 0))
            transparent_img_draw = ImageDraw.Draw(transparent_img)

            if shape.name == 'Triangle':
                transparent_img_draw.polygon(shape.parameters['coordinates'], fill=shape.colour)
                initial_transp = Image.alpha_composite(initial_transp, transparent_img)

        base_image_and_shape.img = initial_transp.convert('RGB')

        return base_image_and_shape


    def compute_distance_between_two_images(self, initial_img, image_to_compare):
        total_distance = np.sum(np.sqrt(pow(initial_img[:,:,0] - image_to_compare[:,:,0], 2) +
                                        pow(initial_img[:,:,1] - image_to_compare[:,:,1], 2) +
                                        pow(initial_img[:,:,2] - image_to_compare[:,:,2], 2)))

        return total_distance / (self.height * self.width)

    def do_it_with_algo_gen(self):
        base_img = ImageAndShapes(Image.new('RGB', self.initial_img.size, tuple(self.mean_colour_original)), [])
        base_img_lab = cv2.cvtColor(np.asarray(base_img.img, dtype='float32') / 255, cv2.COLOR_RGB2Lab)

        distance = self.compute_distance_between_two_images(self.lab_array_initial, base_img_lab)

        nb_iterations = 0

        new_image = deepcopy(base_img)

        for i in range(10):
            print(f'Starting iteration {nb_iterations+1}')
            one_shape = self.generate_shape()
            new_image.shapes.append(one_shape)
            new_image = self.hill_climbing(one_shape, new_image, base_img_lab)

        breakpoint()

        # TODO:
        # Take the n closest pictures
        # Do an AlgoGen stuff (polygons as genes)
        # Do another pass with the output image as a base


    def hill_climbing(self, shape, image, last_image):
        """
            Hill climbing are fun.
        """

        new_img = self.put_shapes_in_image(image)
        new_img_lab = cv2.cvtColor(np.asarray(new_img.img, dtype='float32') / 255, cv2.COLOR_RGB2Lab)

        old_distance = self.compute_distance_between_two_images(self.lab_array_initial, last_image)

        new_distance = self.compute_distance_between_two_images(self.lab_array_initial, new_img_lab)

        while new_distance > old_distance:
            # generate 5 new shapes
            pass
        breakpoint()


    def dst(self, c1, c2):
        return sqrt(pow(c2[0] - c1[0], 2) + pow(c2[1] - c1[1], 2) + pow(c2[2] - c1[2], 2))

def main(path: str):

    image_shaper = ImageShaper(path)
    image_shaper.do_it_with_algo_gen()
    # RGB [0, 255]

    # img = Image.open(path).convert('RGB')
    # # RGB [0, 1]
    # img_rgb_data = np.asarray(img) / 255

    # img_lab_data = color.rgb2lab(img_rgb_data)

    # mean_pixel_value = get_mean_colour(img_lab_data)

    # mean_colour_img = get_mean_colour_array(mean_pixel_value, img.width, img.height, True)



if __name__ == '__main__':
    path = 'test.jpg'
    main(path)
