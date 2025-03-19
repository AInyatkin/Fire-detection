import gradio as gr
from ultralytics import YOLO
import cv2


model = YOLO('best.pt')


def process_video(video_file, confidence_threshold=0.5):

    print(f"Начало обработки видео: {video_file}")

    try:
        video = cv2.VideoCapture(video_file)
        if not video.isOpened():
            raise Exception("Не удалось открыть видеофайл.")
    except Exception as e:
        print(f"Ошибка при открытии видео: {e}")
        return "Не удалось открыть видеофайл."

    detected_frames = []

    frame_rate = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / frame_rate

    for second in range(int(duration)):
        video.set(cv2.CAP_PROP_POS_FRAMES, second * frame_rate)
        ret, frame = video.read()

        if not ret:
            print(f"Не удалось прочитать кадр на {second} секунде. Прекращение обработки.")
            break

        results = model(frame, conf=confidence_threshold)
        detections = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]
                detections.append({'box': [x1, y1, x2, y2], 'confidence': confidence, 'label': label})

        if detections:
            for det in detections:
                x1, y1, x2, y2 = det['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['label']} {det['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frames.append(frame_rgb)

    video.release()

    print("Обработка завершена.")
    return detected_frames


iface = gr.Interface(
    fn=process_video,
    inputs=[gr.Video(), gr.Slider(minimum=0, maximum=1, value=0.5, label="Удовлетворительная надёжность")],
    outputs=gr.Gallery(label="Обнаруженные объекты"),
    title="Объекты детекции пожаров",
    description="Загрузите видео а наша программа, "
                "вернёт обнаруженные пожары с вашего видео."
)

iface.launch()