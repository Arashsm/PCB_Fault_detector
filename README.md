The rep for PCB Fault Detection.
YOLO8 has been trained using https://www.kaggle.com/datasets/akhatova/pcb-defects/data dataset.
An app created with  these endpoints 
 /predict : Accept image data and confidence limit, return predictions
(bounding boxes and labels) as structured data.
 /visualize : Accept image data and confidence limit, return image with
drawn bounding boxes and label annotations.
The prediction results are saved in the database.
