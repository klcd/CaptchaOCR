import tensorflow as tf


def encode_single_sample(img_path, label, img_height, img_width, char_to_num):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers

    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}
