import tensorflow as tf
from matplotlib import pyplot as plt

img_list = [".//plj.jpg", ".//plj2.jpg", ".//plj3.jpg", ".//plj4.jpg"]

# Takes in all the strings within our original image list
# and creates a queue of these filenames
filename_q = tf.train.string_input_producer(img_list)

# Read an entire image file
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    # A session is multi-threaded an we want these multiple threads
    # to read our files

    # Coordinate the loading of images
    # makes working with multiple threads and queues very convinient
    coord = tf.train.Coordinator()

    # Q's are very convinient way to compute tensors asynchronously
    # using multiple threads
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_list = []

    for i in range(len(img_list)):
        # Returns a tuple
        # First field is filename that we'll ignore
        # Second field is the actual image file
        _, image_file = image_reader.read(filename_q)

        # Assuming compression format JPEG, return the tensor representation of the image,
        # which we can use in training
        image = tf.image.decode_jpeg(image_file)

        # Get a tensor of resized image
        # Remember, image is just another tensor
        image = tf.image.resize_images(image, [1000, 1600])
        image.set_shape([1000, 1600, 3])

        image_array = sess.run(image)
        # print(image_array.shape)
        
        # Converts a numpy array of the shape (1000, 1600, 3) to a tensor of shape (1000, 1600, 3),
        # Used to convert rank R tensors to rank R+1 tensors
        image_tensor = tf.stack(image_array)
        # print(image_tensor)

        # Expand the no. of dimensions this image tensor has before adding it to image_list
        # image_list.append(tf.expand_dims(image_array), 0)

        image_list.append(image_tensor)

    coord.request_stop()
    coord.join(threads)

    # Converts list of 3-D image tensor to 4-D images tensor
    images_tensor = tf.stack(image_list)
    print(images_tensor)

    summary_writer = tf.summary.FileWriter('./m4_example1', sess.graph)
    summary_str = sess.run(tf.summary.image("image", images_tensor))
    summary_writer.add_summary(summary_str)