import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools
import cv2
import time

tf.enable_eager_execution()
# tf.executing_eagerly()

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'content':content_dict, 'style':style_dict}


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  # if np.ndim(tensor)>3:
  #   assert tensor.shape[0] == 1
  #   tensor = tensor[0]
  plt.imshow(tensor[0])
  plt.show()

# load images from a specific path and then resize them
def load_img(path_to_img):
    max_dim = 512
    
    im = cv2.imread(path_to_img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(im, dtype=tf.uint8)
    
    #resize
    shape = im.shape
    long_dim = np.amax(shape)
    scale = max_dim / long_dim

    # map every value from the image to a range between [0,1)
    img=tf.image.convert_image_dtype(img, tf.float32)

    tensor_shape = tf.cast(img.get_shape()[:-1], tf.float32)

    new_shape = tf.cast(tensor_shape * scale, tf.int32)
     
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img

# display images from tensors to numpy
def imshow(image, title=None):
    # if len(image.shape) > 3:
    #     image = tf.squeeze(image, axis=0)

    # image = tf.Session().run(image)[0].astype('uint8')
    # print(image.numpy()[0].astype('uint8'))

    image = image.numpy()[0].astype('uint8')
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

def imshow_tensor(image):
  if(len(image.shape) > 3):
    #removes dimensions of size 1
    image = tf.squeeze(image, axis=0)

  plt.imshow()
  plt.show()

def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

# because the images have float values we need to keep them between 0 and 1
def clip_between_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs):
  style_outputs = outputs['style']
  content_outputs = outputs['content']
  style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                          for name in style_outputs.keys()])
  style_loss *= style_weight / num_style_layers

  content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                            for name in content_outputs.keys()])
  content_loss *= content_weight / num_content_layers
  loss = style_loss + content_loss
  return loss

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  print('grad', grad)
  opt.apply_gradients([(grad, image)])
  print( )
  print(image)
  image.assign(clip_between_0_1(image)) 

  

# main
img_path = 'C:\\Users\\bulzg\\Desktop\\real_img.jpg'
style_img_path = 'C:\\Users\\bulzg\\Desktop\\img\\0.jpg'

content_img = load_img(img_path) 
style_img = load_img(style_img_path)


# get vgg19 model and see the layers
# top is the fully connectec layers which we would not use
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
print()
for layer in vgg.layers:
  print(layer.name)

"""
We could replace all the max_pooling operations by average pooling
Replacing the max-pooling operation by average pooling improves the
gradient flow and one obtains slightly more appealing results
"""

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_img*255)

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_img))

style_results = results['style']

print('Styles:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())


style_targets = extractor(style_img)['style']
content_targets = extractor(content_img)['content']


""" 
create the optimised image
to make things a bit more easy we can
initialise it eaither with random noise
that will be corrected incrementaly 
either with the initial image (this approach is faster because it saves us time)
"""
image = tf.Variable(content_img)

opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# To optimize this, use a weighted combination of the two losses to get the total loss
# style_weight can be changer to 1e-2, but for my case 1e-1 gived best results
style_weight=1e-1
content_weight=1e4

# for my case the optimum was 1500, but it can vary from 800-2000
# depending on how much you want to integrate the style
epochs = 10
steps_per_epoch = 150

start = time.time()

# imshow(image)
step = 0
for i in range(epochs):
  for j in range(steps_per_epoch):
    step += 1
    train_step(image)
    print("Train step: ", step)

end = time.time()
print("It took: ", (end-start), 'seconds')
tensor_to_image(image)