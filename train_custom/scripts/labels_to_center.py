import os
import numpy as np
import pickle
import click
import cv2


@click.command()
@click.option('--labels_path', help='Path to labels from yolo')
@click.option('--pickle_path', help='Path to pickle file')
def labels_to_center(labels_path, pickle_path):

    image_path = os.path.dirname(os.path.abspath(labels_path))
    # results_path = os.path.join(os.path.dirname(os.path.abspath(labels_path)),'results')

    data = {}
    # results_file = open(os.path.join(results_path,'results.txt'),"a+")
    for label in os.listdir(labels_path):
        centers = []

        img_name = label.split(".t")[0] + '.png'
        img = cv2.imread(os.path.join(image_path, img_name))

        h, w, _ = img.shape
        file = open(os.path.join(labels_path, label), 'r')

        label_data = file.readlines()

        for detections in label_data:

            det = np.array(detections.split(" ")).astype(np.float64)
            x_center = int(det[1] * w)
            y_center = int(det[2] * h)
            w_det = int(det[3] * w)
            h_det = int(det[4] * h)

            # x_1 = int(x_center-(w_det/2))
            # x_2 = int(x_center + (w_det / 2))
            # y_1 = int(y_center + (h_det / 2))
            # y_2 = int(y_center - (h_det / 2))
            # score = det[-1]

            # results_file.write(f"{img_name},{x_1},{y_1},{x_2},{y_2},{score}\n")
            center = (x_center, y_center)
            centers.append([x_center, y_center])

            # img = cv2.rectangle(img,point1,point2,(255,0,0),3)

            # img = cv2.circle(img,center,3,(0,255,0),-1)
        if len(img_name.split(".")[1]) == 9:
            data[img_name] = centers

    # results_file.close()

        # cv2.imshow("window",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # data[img_name] = centers

    pickle.dump(data, open(os.path.join(pickle_path, "detection_front.pickle"), "wb"))


if __name__ == '__main__':
    labels_to_center()
