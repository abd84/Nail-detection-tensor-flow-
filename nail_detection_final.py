import tensorflow as tf
import numpy as np
import cv2
from imutils.video import WebcamVideoStream
import find_finger as ff

args = {
    "model": "/Users/abdullah/Desktop/Jupyter/nail detection/frozen_inference_graph.pb",
    "labels": "./record/classes.pbtxt",
    "num_classes": 1,
    "min_confidence": 0.6
}

COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))

if __name__ == '__main__':
    detection_graph = tf.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(args["model"], "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            boxes_tensor = detection_graph.get_tensor_by_name("detection_boxes:0")
            scores_tensor = detection_graph.get_tensor_by_name("detection_scores:0")
            classes_tensor = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")

            vs = WebcamVideoStream(src=0).start()
            while True:
                frame = vs.read()
                if frame is None:
                    continue
                frame = cv2.flip(frame, 1)
                image = frame
                (H, W) = image.shape[:2]
                output = image.copy()
                
                img_ff, bin_mask, res = ff.find_hand_old(image.copy())
                image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)

                (boxes, scores, labels, N) = sess.run(
                    [boxes_tensor, scores_tensor, classes_tensor, num_detections],
                    feed_dict={image_tensor: image}
                )
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                labels = np.squeeze(labels)
                box_mid = (0, 0)

                for (box, score, label) in zip(boxes, scores, labels):
                    if score < args["min_confidence"]:
                        continue
                    (startY, startX, endY, endX) = box
                    startX = int(startX * W)
                    startY = int(startY * H)
                    endX = int(endX * W)
                    endY = int(endY * H)
                    X_mid = startX + int(abs(endX - startX) / 2)
                    Y_mid = startY + int(abs(endY - startY) / 2)
                    box_mid = (X_mid, Y_mid)
                    label_name = 'nail'
                    idx = 0
                    label = "{}: {:.2f}".format(label_name, score)
                    cv2.rectangle(output, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(output, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)

                if box_mid == (0, 0):
                    cv2.putText(output, 'Nothing', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                else:
                    cv2.putText(output, 'Nail Detected', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

                cv2.imshow("Output", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

            vs.stop()
