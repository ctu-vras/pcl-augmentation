# import matplotlib.pyplot as plt
import numpy as np
from skimage.util import img_as_ubyte
from skimage.morphology import closing
from skimage.morphology import rectangle
from PIL import Image

PIPELINE = [5, 0, 1, 3, 4, 2]
CLASS_INDEX = [1, 40, 44, 48]
DISK_SIZE = 3
# Road, Background, Car, Bikes, Road blocks, Humans
RGB_CLASS = np.array([[255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 0, 255], [123, 123, 123], [0, 255, 0], [0, 0, 0], [255, 192, 203]])

def create_image(labels, filename):
    """
    Function, which create Image of classes.
    :param labels: numpy 2D array with classes
    :param filename: string, name of the image
    """

    columns = len(labels[0])
    lines = len(labels)
    rgb = np.zeros(3 * columns * lines).reshape(lines, columns, 3)

    for i in range(lines):
        for j in range(columns):
            labels_idx = int(labels[i][j])
            rgb[i][j] = RGB_CLASS[labels_idx]

    rgb = np.uint8(rgb)
    img = Image.fromarray(rgb, 'RGB')
    img.save(filename)
    # img.show()
    return


def class_closing(original_label):
    """
    Function, which uses closing to the FoV
    :param original_label: numpy 2D array, label FoV made from point-cloud
    :return: numpy 2D array, closed label FoV
    """
    class_mask = original_label.copy()

    for row in range(original_label.shape[0]):
        for column in range(original_label.shape[1]):
            if original_label[row][column] in CLASS_INDEX:
                class_mask[row][column] = 1
            else:
                class_mask[row][column] = 0

    mask_phantom = img_as_ubyte(class_mask)
    selem = rectangle(5, 3)
    closed = closing(mask_phantom, selem)

    return closed


def smooth_out(original_train, original_label):
    """
    Function, which closes FoV for eliminating black spots in objects
    :param original_train: numpy 2D array, original FoV with shape (NUMROW, NUMCOLUMN, 3) in 1.channel distance, 2.channel intensity, 3.channel code if pixel represent any point.
    :param original_label: numpy 2D array, original FoV with shape (NUMROW, NUMCOLUMN). Pixels code label of nearest point.
    :return: numpy 2D array, closed original_train and original_label
    """
    train = original_train.copy()
    label = original_label.copy()

    closed_layer = class_closing(original_label)    # check value in array
    for row in range(original_label.shape[0]):              # compare closing results
        for column in range(original_label.shape[1]):
            if (closed_layer[row][column] == 255 and label[row][column] in CLASS_INDEX) or closed_layer[row][column] == 0:
                continue
            else:
                neighbors = 0
                sum_distance = 0
                for drow in range(-2, 3):
                    for dcolumn in range(-1, 2):
                        if -1 < drow + row < original_label.shape[0] and -1 < dcolumn + column < original_label.shape[1] and original_label[drow + row][dcolumn + column] in CLASS_INDEX:
                            neighbors += 1
                            sum_distance += original_train[row + drow][column + dcolumn][0]
                if neighbors == 0:
                    print('No neighbor?!')
                    print('ROW:', row, 'COLUMN:', column)
                    label[row][column] = 1
                else:
                    train[row][column][0] = sum_distance/neighbors
                    label[row][column] = 1
                    if train[row][column][0] == 0:
                        print('ERROR in function: smooth_out - sum_distance == 0')

    return train, label