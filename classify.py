import sys
import caffe
from caffe.proto import caffe_pb2
import numpy
import cv2
import matplotlib.pyplot as plt
net_path='deploy.prototxt'
model_path='caffemodel/mymodel.h5'
mean_path='mean.npy'

cifar_map={
     0:Red ball,
     1:Yellow ball,
     2:Blue ball,
     3:Field
}

mean_blob=caffe_pb2.BlobProto()

with open(mean_path) as f:
    mean_blob.ParseFromString(f.read())

    mean_array=numpy.asarray(
        mean_blob.data,
        dtype=numpy.float32
    ).reshape(
        (mean_blob.channels, mean_blob.height, mean_blob.width)
    )

    classifier=caffe.Classifier(
        net_path,
        model_path,
        mean=mean_array,
        raw_scale=255
    )

    image=caffe.io.load_image(sys.argv[1])

    predictions=classifier.predict([image],oversample=False)
    answer=numpy.argmax(predictions)

    for index, prediction in enumerate(predictions[0]):
        print(str(index)+"("+cifar_map[index]+"):").ljust(15)+str(prediction)
    print("This image is ["+cifar_map[answer]+"]")

i = 0
Input = cv2.imread(sys.argv[1])
Output = cv2.imread("Class/" + cifar_map[answer] + ".jpg")

while i == 0:
    cv2.imshow("Input", Input)
    cv2.imshow("Output", Output)
    keycode = cv2.waitKey(0)
    if keycode == 27:
        cv2.destroyAllWindows()
        i = 1
