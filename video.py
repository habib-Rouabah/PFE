import cv2
import torch
from PIL import Image

# Charger le modèle YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Charger la vidéo
video = cv2.VideoCapture('yolov5-master/kkk/kkkkkm/lllll.mp4')

# Définir les paramètres de sortie de la vidéo
output_file = 'path/to/output.avi'
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Convertir la frame en format PIL
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Effectuer une détection d'objets avec YOLOv5
    results = model(frame)
    color = (0, 255, 0)
    thickness = 2
    tensor = results.xyxy[0]
    mask = tensor[:, -1] == 0
    rows_with_zero = tensor[mask]
    resu = rows_with_zero[:, :4]
    b = resu.numpy()
    b.flatten()
    #print(b)
    w = b[0][2] - b[0][0]
    h = b[0][3] - b[0][1]
    r = w / h
    for i in resu:
        start_point = (int(i[0]), int(i[1]))
        end_point = (int(i[2]), int(i[3]))
        cv2.rectangle(frame, start_point, end_point, color, thickness)

        #Afficher le texte "Fall" ou "No Fall" à côté de la boîte englobante
        text = "Fall" if r > 0.55 else "No Fall"
        text_position = (int(i[0]), int(i[1]) - 10)
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

    # Afficher la frame avec les boîtes englobantes et le texte
    cv2.imshow('Video', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video.release()
out.release()
cv2.destroyAllWindows()

