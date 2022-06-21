import numpy as np
from skimage.util import img_as_ubyte
from skimage.morphology import closing
from skimage.morphology import rectangle
from PIL import Image
import time


def class_closing(original_label):
    """
    Function, which uses closing to the FoV
    :param original_label: numpy 2D array, label FoV made from point-cloud
    :return: numpy 2D array, closed label FoV
    """
    class_mask = original_label.copy()

    class_mask = np.clip(class_mask, 0, 1)

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

    closed_layer = class_closing(original_label)

    for row in range(original_label.shape[0]):
        for column in range(original_label.shape[1]):
            if (closed_layer[row][column] == 255 and label[row][column] == 1) or closed_layer[row][
                column] == 0:
                continue
            else:
                neighbors = 0
                sum_distance = 0
                for drow in range(-2, 3):
                    for dcolumn in range(-1, 2):
                        if -1 < drow + row < original_label.shape[0] and -1 < dcolumn + column < original_label.shape[
                            1] and original_label[drow + row][dcolumn + column] == 1:
                            neighbors += 1
                            sum_distance += original_train[row + drow][column + dcolumn]
                if neighbors == 0:
                    print('No neighbor?!')
                    print('ROW:', row, 'COLUMN:', column)
                    label[row][column] = 1
                else:
                    train[row][column] = sum_distance / neighbors
                    label[row][column] = 1
                    if train[row][column] == 0:
                        print('ERROR in function: smooth_out - sum_distance == 0')

    # print(f'computing {time.time() - start:.02f}')

    return train, label
