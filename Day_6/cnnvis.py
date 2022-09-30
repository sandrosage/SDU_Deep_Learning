# Take from
# https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html


import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.preprocessing.image import save_img
from keras.applications import vgg16
from keras import backend as K

vgg = vgg16.VGG16(weights='imagenet', include_top=False)
vgg.summary()

layer_dict = dict([(layer.name, layer) for layer in vgg.layers[1:]])
layer_dict

input_img = vgg.inputs[0]

layer_name = "block4_conv1"
filter_index = 16  # any filter of that layer

layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, filter_index])

grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])

output_dim = (412, 412)

input_img_data = np.random.random((1, output_dim[0], output_dim[1], 3))
input_img_data = (input_img_data - 0.5) * 20 + 128

for _ in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * 1

input_img_data -= input_img_data.mean()
input_img_data /= (input_img_data.std() + K.epsilon())
input_img_data *= 0.25

# clip to [0, 1]
input_img_data += 0.5
input_img_data = np.clip(input_img_data, 0, 1)

# convert to RGB array
input_img_data *= 255
input_img_data = np.clip(input_img_data, 0, 255).astype('uint8')

def _draw_filters(filters, n=None):
    """Draw the best filters in a nxn grid.
    # Arguments
        filters: A List of generated images and their corresponding losses
                 for each processd filter.
        n: dimension of the grid.
           If none, the largest possible square will be used
    """
    if n is None:
        n = int(np.floor(np.sqrt(len(filters))))

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top n*n filters.
    filters.sort(key=lambda x: x[1], reverse=True)
    filters = filters[:n * n]

    # build a black picture with enough space for
    # e.g. our 8 x 8 filters of size 412 x 412, with a 5px margin in between
    MARGIN = 5
    width = n * output_dim[0] + (n - 1) * MARGIN
    height = n * output_dim[1] + (n - 1) * MARGIN
    stitched_filters = np.zeros((width, height, 3), dtype='uint8')

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, _ = filters[i * n + j]
            width_margin = (output_dim[0] + MARGIN) * i
            height_margin = (output_dim[1] + MARGIN) * j
            stitched_filters[
            width_margin: width_margin + output_dim[0],
            height_margin: height_margin + output_dim[1], :] = img

    # save the result to disk
    save_img('vgg_{0:}_{1:}x{1:}.png'.format(layer_name, n), stitched_filters)


_draw_filters([(input_img_data, 0.1)])
