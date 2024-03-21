import numpy as np
import pathlib
import tensorflow as tf
from tensorflow.train import Example, Features, Feature, BytesList, Int64List
from PIL import Image
import os


cur_dir = pathlib.Path(__file__).parent.resolve()
# X = np.array([], dtype=np.int64)
X = []
y = np.array([], dtype=np.int64)
cnt = 0
for root, dirs, files in os.walk(os.path.join(cur_dir, 'data')):
    cnt += 1
    if cnt == 1000:
        break
    if files:
        i = len(root) - 1
        while root[i].isdigit():
            i -= 1
        label = root[i+1:]
        if not label:
            label = 2000
        print(label)
        for img_idx, img in enumerate(files):
            with Image.open(os.path.join(root, img)) as parsed_img:
                # X = np.append(X, np.asarray(parsed_img), axis=0) if X.shape[0] != 0 else np.asarray(parsed_img)
                # print(np.asarray(parsed_img).shape)
                X.append(tf.io.encode_png(parsed_img).numpy())
                # print(parsed_img.tobytes())
                # print(X.shape)
                y = np.append(y, int(label))
# X = np.array(X)
# y = np.array(y, dtype=np.int64)
# X = X.reshape((-1, 112, 112, 4))
# X = X[..., :3]
# print(X.shape, y.shape)
# X = [tf.io.encode_png(img).numpy() for img in X]
example = Example(features=Features(feature={
    'features': Feature(bytes_list=BytesList(value=X)),
    'labels': Feature(int64_list=Int64List(value=y)) 
}))
serialized_example = example.SerializeToString()
with tf.io.TFRecordWriter(path=os.path.join(cur_dir, "RGB_encoded_data.tfrecord")) as f:
    f.write(serialized_example)
