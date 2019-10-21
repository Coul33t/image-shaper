import random as rn
import numpy as np

from PIL import Image, ImageDraw
from skimage import color
import cv2

from math import sqrt

#TODO:
# Algo gen on the min/max size of poly, and range of colours maybe?

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

        self.mean_colour = self.get_mean_colour(self.initial_img)

    def get_mean_colour(self, img: Image) -> np.ndarray:
        """
            Compute the mean colour from an image.
        """

        mean_pixel_value = np.asarray([0, 0, 0], np.float64)

        all_colours = img.getcolors(img.size[0] * img.size[1])

        for colour in all_colours:
            mean_pixel_value += np.asarray(colour[1]) * colour[0]

        mean_pixel_value = mean_pixel_value / (self.width * self.height)

        return mean_pixel_value.astype('uint8')


    def put_shapes_in_image(self, base_image_and_shape, shapes):
        initial_transp = base_image_and_shape.img.convert('RGBA')

        for i, shape in enumerate(base_image_and_shape.shapes):
            transparent_img = Image.new('RGBA', self.initial_img.size, (255,255,255,0))
            transparent_img_draw = ImageDraw.Draw(transparent_img)

            if shape.name == 'Triangle':
                transparent_img_draw.polygon(shape.parameters, fill=shape.colour)
                initial_transp = Image.alpha_composite(initial_transp, transparent_img)

        return ImageAndShapes(initial_transp.convert('RGB'), shapes)

    def generate_shapes(self, shapes_to_use: list = [], k: int = 100):
        shapes = []
        if not shapes_to_use:
            shapes_to_use = ['Triangle' for x in range(k)]

        for shape in shapes_to_use:

            new_colour = (rn.randint(0, 255),
                          rn.randint(0, 255),
                          rn.randint(0, 255))

            if shape == 'Triangle':
                h = self.rgb_array_initial.shape[0]
                w = self.rgb_array_initial.shape[1]
                coordinates = [rn.randint(0, w), rn.randint(0, h),
                               rn.randint(0, w), rn.randint(0, h),
                               rn.randint(0, w), rn.randint(0, h)]

            shapes.append(Shape(shape, new_colour, {'coordinates': coordinates}))

        return shapes

    def generate_image(self, base_image_and_shape: ImageAndShapes, k: int = 100):
        mean_colour = self.mean_colour
        # Alpha channel
        np.append(mean_colour, 0)
        initial_transp = Image.new('RGBA', base_image_and_shape.img.size, tuple(mean_colour))

        # TODO: check if it's really needed to recreate a new image each time
        # (see https://stackoverflow.com/questions/359706/how-do-you-draw-transparent-polygons-with-python)
        # Redraw the image with the shapes
        for shape in base_image_and_shape.shapes:
            transparent_img = Image.new('RGBA', self.initial_img.size, (255,255,255,0))
            transparent_img_draw = ImageDraw.Draw(transparent_img)
            # Append the alpha channel
            transparent_img_draw.polygon(shape.parameters['coordinates'], fill=tuple(np.append(shape.colour, 127)))
            initial_transp = Image.alpha_composite(initial_transp, transparent_img)

        # initial_transp.convert('RGB').show()
        base_image_and_shape.img = initial_transp.convert('RGB')

    def generate_subimage_one_shape(self, base_image_and_shape: ImageAndShapes, idx: int):
        mean_colour = self.mean_colour
        # Adding the alpha channel
        np.append(mean_colour, 0)

        shape = base_image_and_shape.shapes[idx]

        min_x = min([x for i, x in enumerate(shape.parameters['coordinates']) if i%2 == 0])
        max_x = max([x for i, x in enumerate(shape.parameters['coordinates']) if i%2 == 0])
        min_y = min([x for i, x in enumerate(shape.parameters['coordinates']) if i%2 == 1])
        max_y = max([x for i, x in enumerate(shape.parameters['coordinates']) if i%2 == 1])

        initial_transp = Image.new('RGBA', (max_x - min_x, max_y - min_y), tuple(mean_colour))

        old_coordinates = shape.parameters['coordinates']
        new_coordinates = [0 for i in range(len(old_coordinates))]

        for i in range(len(new_coordinates)):
            if i % 2 == 0:
                new_coordinates[i] = old_coordinates[i] - min_x
                new_coordinates[i]
            else:
                new_coordinates[i] = old_coordinates[i] - min_y

        # Redraw the image with the shape
        transparent_img = Image.new('RGBA', (max_x - min_x, max_y - min_y), (255,255,255,0))
        transparent_img_draw = ImageDraw.Draw(transparent_img)
        # Append the alpha channel
        transparent_img_draw.polygon(new_coordinates, fill=tuple(np.append(shape.colour, 127)))
        initial_transp = Image.alpha_composite(initial_transp, transparent_img)

        sub_img_coord = (min_x, min_y, max_x, max_y)

        # initial_transp.convert('RGB').show()
        return sub_img_coord, initial_transp.convert('RGB')

    def compute_distance(self, image):
        """
            Compute the total distance between two images, after switching them
            to the L*a*b* space.
        """
        # HUGE BOTTLENECK HERE! scikit-image's rgb2lab() is really slow
        # replace with OpenCV one (should be faster)
        # DONE: 3 times faster

        image_lab_data = cv2.cvtColor(np.asarray(image, dtype='float32') / 255, cv2.COLOR_RGB2Lab)
        initial_lab_data = self.lab_array_initial

        total_distance = np.sum(np.sqrt(pow(image_lab_data[:,:,0] - initial_lab_data[:,:,0], 2) +
                                        pow(image_lab_data[:,:,1] - initial_lab_data[:,:,1], 2) +
                                        pow(image_lab_data[:,:,2] - initial_lab_data[:,:,2], 2)))

        return total_distance / (self.height * self.width)

    def compute_distance_between_two_images(self, initial_img, image_to_compare, sub_img_coord):
        sub_initial_img = initial_img[sub_img_coord[1]:sub_img_coord[3], sub_img_coord[0]:sub_img_coord[2]]

        total_distance = np.sum(np.sqrt(pow(sub_initial_img[:,:,0] - image_to_compare[:,:,0], 2) +
                                        pow(sub_initial_img[:,:,1] - image_to_compare[:,:,1], 2) +
                                        pow(sub_initial_img[:,:,2] - image_to_compare[:,:,2], 2)))

        return total_distance / (self.height * self.width)

    def compute_shapes_distance(self, image_and_shapes: ImageAndShapes):
        image_lab_data = self.lab_array_initial

        distances = []

        for i in range(len(image_and_shapes.shapes)):
            sub_img_coord, one_shape_image = self.generate_subimage_one_shape(image_and_shapes, i)
            one_shape_image = cv2.cvtColor(np.asarray(one_shape_image, dtype='float32') / 255, cv2.COLOR_RGB2Lab)
            distances.append(self.compute_distance_between_two_images(image_lab_data, one_shape_image, sub_img_coord))

        return distances

    def do_it_without_algo_gen(self):
        number_of_polygons = 100
        to_keep = 1
        shapes = ['Triangle' for i in range(number_of_polygons)]
        final_image = ImageAndShapes(Image.new('RGB', self.initial_img.size, tuple(self.mean_colour)), [])
        iterations = 0

        print(f'Number of polygons generated each iteration: {number_of_polygons}')
        print(f'Number of shapes to keep each iteration: {to_keep}')
        # TODO: modify this (count occurence of each) when more shapes
        print(f'Shapes to be generated: Triangles')

        while len(final_image.shapes) < 1000:
            print(f'Starting iteration {iterations+1}')
            new_shapes = self.generate_shapes(shapes, number_of_polygons)
            final_image.shapes.extend(new_shapes)
            self.generate_image(final_image, k=number_of_polygons)

            shapes_distances = self.compute_shapes_distance(final_image)

            closest_shapes_idx = []
            for _ in range(iterations + to_keep):
                idx = shapes_distances.index(min(shapes_distances))
                closest_shapes_idx.append(idx)
                shapes_distances[idx] = max(shapes_distances)

            shapes_to_keep = [final_image.shapes[x] for x in closest_shapes_idx]

            final_image.shapes = shapes_to_keep
            self.generate_image(final_image, k=number_of_polygons)
            iterations += 1

        final_image.img.show()


    def do_it_with_algo_gen(self):
        number_of_images = 50
        images = []

        for i in range(number_of_images):
            # MUST put the empty list or else all the lists point to the same
            # one (?????)
            # print(f'Generating base image {i+1}...', end='')
            images.append(ImageAndShapes(Image.new('RGB', self.initial_img.size, tuple(self.mean_colour)), []))
            # print('Done.')

        # Pretty long for 10 shapes
        for i in range(number_of_images):
            # print(f'Generating first iteration of image {i+1}...', end=' ')
            self.generate_image(images[i], k=10)
            # print('Done.')


        distances = []
        for i in range(number_of_images):
            # print(f'Computing distance from original image to image {i+1}...', end=' ')
            distances.append(self.compute_distance(images[i].img))
            # print('Done.')

        nb_iterations = 0
        lowest_distance = [min(distances), nb_iterations+1]

        for i in range(10):
        #while min(distances) > 10:
            print(f'Starting iteration {nb_iterations+1}')
            new_shapes = self.algotum_geneticum(images, distances)

            #TODO: redraw the whole image for now, the old shapes are opaques
            # when adding new ones (eh that is actually a nice hint about why
            # it currently sucks)

            for i in range(number_of_images):
                # print(f'Generating new image {i+1}...', end=' ')
                images[i] = self.put_shapes_in_image(images[i], new_shapes[i])
                # print('Done.')

            distances = []

            for i in range(number_of_images):
                # print(f'Computing distance from original image to image {i+1}...', end=' ')
                distances.append(self.compute_distance(images[i].img))
                # print('Done.')

            print(f'All distances:', end=' ')
            for dst in distances:
                print(f'{dst:.2f}', end='|')

            print('')
            print(f'Lowest distance: {min(distances):.2f}')
            print(f'Average of distances: {sum(distances)/len(distances):.2f}')

            nb_iterations += 1

            if min(distances) <= lowest_distance[0]:
                lowest_distance[0] = min(distances)
                lowest_distance[1] = nb_iterations + 1

            print(f'Lowest distance so far: {lowest_distance[0]:.2f} at iteration n°{lowest_distance[1]}')

        print(f'One distance is less than 10')



    def algotum_geneticum(self, images, distances):
        """
            Genetic Algorithms are fun.
        """

        mutation_chances = 0.05

        number_of_closest = 3

        if number_of_closest > len(images):
            number_of_closest = len(images) - 1
        closest_images_idx = []

        for i in range(number_of_closest):
            idx = distances.index(min(distances))
            closest_images_idx.append(idx)
            distances[idx] = max(distances)

        genes = [images[x].shapes for x in range(number_of_closest)]
        new_genes = []

        # Cross breeding (fuckin' inbreds)
        for i in range(len(images)):
            sample_genes = []
            for j in range(len(genes[0])):
                rand = rn.randint(0, len(genes) - 1)
                sample_genes.append(genes[rand][j])
            new_genes.append(sample_genes)

        # Mutation (u fookin wot mate)
        # Not used for the moment as I want to see how it goes without it
        # for i in range(len(images)):
        #     for j in range(len(new_genes[0])):
        #         rand = rn.random()
        #         if rand < mutation_chances:
        #             new_genes[i][j]

        return new_genes

    def dst(self, c1, c2):
        return sqrt(pow(c2[0] - c1[0], 2) + pow(c2[1] - c1[1], 2) + pow(c2[2] - c1[2], 2))

def main(path: str):

    image_shaper = ImageShaper(path)

    image_shaper.do_it_without_algo_gen()
    # image_shaper.do_it_with_algo_gen()
    # RGB [0, 255]

    # img = Image.open(path).convert('RGB')
    # # RGB [0, 1]
    # img_rgb_data = np.asarray(img) / 255

    # img_lab_data = color.rgb2lab(img_rgb_data)

    # mean_pixel_value = get_mean_colour(img_lab_data)

    # mean_colour_img = get_mean_colour_array(mean_pixel_value, img.width, img.height, True)

def test_opencv_scikit_image_time():
    import time

    im = Image.open('test.png').convert('RGB')

    t = time.time()
    cv2.cvtColor(np.asarray(im, dtype='float32') / 255, cv2.COLOR_RGB2Lab)
    t_cv = time.time() - t

    t = time.time()
    color.rgb2lab(np.asarray(im) / 255)
    t_si = time.time() - t

    print(f'OpenCV rgb2lab time: {t_cv}')
    print(f'scikit-image rgb2lab time: {t_si}')

if __name__ == '__main__':
    path = 'test2.png'
    main(path)
