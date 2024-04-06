import numpy as np
import pathlib
import tensorflow as tf
from tensorflow.train import Example, Features, Feature, BytesList, Int64List
from PIL import Image
import os


cur_dir = pathlib.Path(__file__).parent.resolve()
# using regular list instead of np array or tensor, because appending is faster
X = []
y = []
cnt = 0
for root, dirs, files in os.walk(os.path.join(cur_dir, 'second_dataset')):
    cnt += 1
    # if cnt == 1000:
    #     break
    if files:
        i = len(root) - 1
        while root[i].isdigit():
            i -= 1
        label = root[i+1:]
        print(label)
        for img_idx, img in enumerate(files):
            with Image.open(os.path.join(root, img)) as parsed_img:
                parsed_img = np.asarray(parsed_img)[..., :1]
                X.append(tf.io.encode_png(parsed_img).numpy())
                y.append(int(label) + 1)
            break
            
example = Example(features=Features(feature={
    'features': Feature(bytes_list=BytesList(value=X)),
    'labels': Feature(int64_list=Int64List(value=y)) 
}))
serialized_example = example.SerializeToString()
with tf.io.TFRecordWriter(path=os.path.join(cur_dir, "second_dataset_test.tfrecord")) as f:
    f.write(serialized_example)
