import random as rn
import numpy as np

from PIL import Image, ImageDraw
from skimage import color

from math import sqrt


class Shape:
    def __init__(self, name: str, parameters: dict):
        self.name = name
        self.parameters = parameters

class ImageShaper:
    def __init__(self, path_to_initial_img):
        self.shape = 'Triangle'
        # Each shape is a gene
        self.shapes_parameters = []
        self.initial_img = Image.open(path).convert('RGB')
        self.rgb_array_initial = np.asarray(self.initial_img)
        self.lab_array_initial = color.rgb2lab(self.rgb_array_initial)
        self.shaped_img = self.initial_img.copy().convert('RGBA')

        self.width = self.rgb_array_initial.shape[1]
        self.height = self.rgb_array_initial.shape[0]

        # Min size for the shapes (idk how to use this but I'll figure it out)
        self.min_size = (int(self.rgb_array_initial.shape[0] / 100),
                         int(self.rgb_array_initial.shape[1] / 100))

    def get_mean_colour(self, img: np.ndarray) -> np.ndarray:
        """
            Compute the mean colour from a nump array representing the image.
        """

        mean_pixel_value = np.asarray([0, 0, 0], np.float64)

        for i in range(self.height):
            for j in range(self.width):
                mean_pixel_value += img[i, j]

        mean_pixel_value = mean_pixel_value / (self.width * self.height)

        return mean_pixel_value


    def generate_image(self, k=100):
        mean_colour = self.get_mean_colour(self.rgb_array_initial).astype('uint8')
        np.append(mean_colour, 0)
        # Alpha channel
        initial_transp = Image.new('RGBA', self.initial_img.size, tuple(mean_colour))
        # Generate an image containing k shapes
        for i in range(k):
            transparent_img = Image.new('RGBA', self.initial_img.size, (255,255,255,0))
            transparent_img_draw = ImageDraw.Draw(transparent_img)
            # Last one is alpha
            new_colour = (rn.randint(0, 255),
                          rn.randint(0, 255),
                          rn.randint(0, 255),
                          127)

            if self.shape == 'Triangle':
                h = self.rgb_array_initial.shape[0]
                w = self.rgb_array_initial.shape[1]
                coordinates = [rn.randint(0, w), rn.randint(0, h),
                               rn.randint(0, w), rn.randint(0, h),
                               rn.randint(0, w), rn.randint(0, h)]

                self.shapes_parameters.append(Shape(self.shape, coordinates))

            transparent_img_draw.polygon(coordinates, fill=new_colour)
            initial_transp = Image.alpha_composite(initial_transp, transparent_img, )

        # initial_transp.convert('RGB').show()
        return initial_transp.convert('RGB')

    def compute_distance(self, image):
        """
            Compute the total distance between two images in L*a*b* space.
        """
        image_lab_data = color.rgb2lab(np.asarray(image) / 255)
        initial_lab_data = color.rgb2lab(np.asarray(self.initial_img) / 255)

        total_distance = np.sum(np.sqrt(pow(image_lab_data[:,:,0] - initial_lab_data[:,:,0], 2) +
                                        pow(image_lab_data[:,:,1] - initial_lab_data[:,:,1], 2) +
                                        pow(image_lab_data[:,:,2] - initial_lab_data[:,:,2], 2)))

        return total_distance / (self.height * self.width)

    def do_it(self):
        number_of_images = 5
        images = []

        for i in range(number_of_images):
            print(f'Generating image {i+1}...', end=' ')
            images.append(self.generate_image(k=10))
            print('Done.')


        # Yeah I know double loop but I want to keep them separated for now
        distances = []
        for i in range(number_of_images):
            distances.append(self.compute_distance(images[i]))

        breakpoint()
        # TODO:
        # Take the n closest pictures
        # Do an AlgoGen stuff (polygons as genes)
        # Do another pass with the output image as a base


    def dst(self, c1, c2):
        return sqrt(pow(c2[0] - c1[0], 2) + pow(c2[1] - c1[1], 2) + pow(c2[2] - c1[2], 2))

def main(path: str):

    image_shaper = ImageShaper(path)
    image_shaper.do_it()
    # RGB [0, 255]
    img = Image.open(path).convert('RGB')
    # RGB [0, 1]
    img_rgb_data = np.asarray(img) / 255

    img_lab_data = color.rgb2lab(img_rgb_data)

    mean_pixel_value = get_mean_colour(img_lab_data)

    mean_colour_img = get_mean_colour_array(mean_pixel_value, img.width, img.height, True)



if __name__ == '__main__':
    path = 'test.jpg'
    main(path)
