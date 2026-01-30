import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from utils import rescale, send_to_kafka
from config.params import camera_configs
from rtsp_stream import RTSPStream
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,  # hoặc DEBUG nếu bạn muốn xem chi tiết
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Khởi tạo mô hình YOLOv11 và OSNet một lần ở cấp độ module
try:
    YOLO_MODEL = YOLO("./weights/best_fire.onnx", task="detect")
    logging.info(f"Successfully loaded YOLOv11 model globally with {YOLO_MODEL.device}")
except Exception as e:
    logging.error(f"Error loading global YOLO model: {e}")
    raise

class CustomerTracker:
    def __init__(self, store_id, camera_id, output_path, size=480, show_video=True, send_api = False, video_path=None):
        self.show_video = True
        self.video_path = video_path
        self.send_api = send_api
        if store_id not in camera_configs:
            raise ValueError(f"store_id {store_id} not found")
        store_cfg = camera_configs[store_id]
        if camera_id not in store_cfg["cameras"]:
            raise ValueError(f"camera_id {camera_id} not found in store {store_id}")

        config = store_cfg["cameras"][camera_id]
        self.camera_id = camera_id
        self.output_path = output_path
        self.size = size

        self.current_frame = 0
        self.model = YOLO_MODEL
        self.class_name = ["Fire","Smoke"]
        # ==== Fire alert config ====
        self.counter_fire_frame = 0
        self.reset_fire_frame = 0
        self.fire_confs = []
        self.max_fire_conf = 0.0
        self.fire_alerted = False   # tránh spam
        self.fire_warring_time = 10
        self.reset_fire_time = 20
        # ==== Smoke alert config ====
        self.counter_smoke_frame = 0
        self.reset_smoke_frame = 0
        self.smoke_confs = []
        self.max_smoke_conf = 0.0
        self.smoke_alerted = False   # tránh spam
        self.smoke_warring_time = 10
        self.reset_smoke_time = 20

        

        # ======== CHỌN NGUỒN VIDEO ========
        if self.video_path and os.path.exists(self.video_path):
            logging.info(f"Camera {self.camera_id}: sử dụng video file {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
            self.is_video_file = True
        else:
            logging.info(f"Camera {self.camera_id}: sử dụng RTSP stream")
            self.rtsp_stream = RTSPStream(config["rtsp_url"], self.camera_id)
            self.rtsp_stream.start()
            self.cap = self.rtsp_stream.cap
            self.is_video_file = False
       # ======== Lấy thông tin video ========
        time.sleep(0.5)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 15
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        logging.info(f"Cam {self.camera_id}: fps={self.fps}, size=({self.width}x{self.height})")

        # ======== Output video ========
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_dir = os.path.dirname(output_path) if '/' in output_path else './output'
        os.makedirs(out_dir, exist_ok=True)
        self.heatmap_video_path = os.path.join(out_dir, f'heatmap_cam{self.camera_id}_4.mp4')
        self.video_writer = cv2.VideoWriter(self.heatmap_video_path, fourcc, self.fps, (self.width, self.height))



        from collections import deque

        # buffer 30s
        self.record_seconds = 30
        self.buffer_size = self.fps * self.record_seconds
        self.frame_buffer = deque(maxlen=self.buffer_size)

        # record state
        self.recording = False
        self.record_start_time = None
        self.record_writer = None

        # snapshot
        self.snapshot_taken = False
        self.snapshot_10s_taken = False
    def start_recording(self, event_type):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"./alerts/{event_type}"
        os.makedirs(out_dir, exist_ok=True)

        video_path = f"{out_dir}/{event_type}_cam{self.camera_id}_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.record_writer = cv2.VideoWriter(
            video_path, fourcc, self.fps, (self.width, self.height)
        )

        # ghi buffer cũ (trước alert)
        for f in self.frame_buffer:
            self.record_writer.write(cv2.resize(f, (self.width, self.height)))

        self.recording = True
        self.record_start_time = time.time()
        self.snapshot_taken = False
        self.snapshot_10s_taken = False

        logging.info(f"🎥 Start recording {event_type} video: {video_path}")
    def stop_recording(self):
        if self.record_writer:
            self.record_writer.release()
        self.record_writer = None
        self.recording = False
        logging.info("🎬 Stop recording alert video")
    def save_snapshot(self, frame, event_type, suffix):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"./alerts/{event_type}"
        os.makedirs(out_dir, exist_ok=True)

        img_path = f"{out_dir}/{event_type}_cam{self.camera_id}_{suffix}_{ts}.jpg"
        cv2.imwrite(img_path, frame)
        logging.info(f"🖼 Saved snapshot: {img_path}")

    def process_frame(self, frame):
        self.current_frame += 1
        frame_resize = cv2.resize(frame, (self.size, self.size))
        annotated_frame = frame.copy()
        h, w, _ = annotated_frame.shape
        have_fire = False
        have_smoke = False
        warning_text = f"SENDED MESS TO KAFKA !!"
        box_color = (0,0,255)
        results = self.model.predict(
            frame_resize,
            conf=0.4,
            iou=0.6,
            verbose=False
        )

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                cls = int(box.cls[0].cpu())
                conf = box.conf[0].cpu().item()
                x1, y1, x2, y2 = box.xyxy[0].cpu()
                x_min, y_min, x_max, y_max = rescale(
                    annotated_frame, self.size, x1, y1, x2, y2
                )
                if cls == 1:
                    have_smoke = True
                    self.smoke_confs.append(conf)
                    self.max_smoke_conf = max(self.max_smoke_conf, conf)
                    box_color = (255,0,0)
                if cls == 0:
                    have_fire = True
                        # ====== UPDATE STATS ======
                    self.fire_confs.append(conf)
                    self.max_fire_conf = max(self.max_fire_conf, conf)
                    box_color = (0,0,255)
                
                # ====== DRAW BOX ======
                cv2.rectangle(
                    annotated_frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    box_color,
                    2
                )
                cv2.putText(
                    annotated_frame,
                    f"{self.class_name[cls]} {conf:.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                    )
                
        # ====== TIME LOGIC ======
        if have_fire:
            self.counter_fire_frame += 1
            self.reset_fire_frame = 0
        else:
            self.reset_fire_frame += 1
            if self.reset_fire_frame > self.reset_fire_time and self.counter_fire_frame > 0:
                self.reset_fire_state()

        if have_smoke:
            self.counter_smoke_frame += 1
            self.reset_smoke_frame = 0
        else:
            self.reset_smoke_frame += 1
            if self.reset_smoke_frame > self.reset_smoke_time and self.counter_smoke_frame > 0:
                self.reset_smoke_state()

        # ====== Mess Fire LERT ======
        if self.counter_fire_frame > self.fire_warring_time:
            avg_fire_conf = np.mean(self.fire_confs) if self.fire_confs else 0
            if self.show_video:
                cv2.putText(
                    annotated_frame,
                    f"Warring:  Fire Certainty {min(100,(avg_fire_conf *100) + (self.counter_fire_frame - self.fire_warring_time)*0.2):.0f}%",
                    (10,int(h*0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            if (
                self.counter_fire_frame >= 20 and
                avg_fire_conf >= 0.5 and
                self.max_fire_conf >= 0.6 and
                not self.fire_alerted
            ):
                
                self.fire_alerted = True
                if self.send_api:
                    send_to_kafka(self.box_id, self.cam_id, "fire", self.max_fire_conf)
                # start record
                self.start_recording("fire")
                # snapshot ngay
                self.save_snapshot(annotated_frame, "fire", "now")
                # reset timer snapshot 10s
                self.fire_alert_time = time.time()

                logging.warning(
                    f"🔥 FIRE ALERT Cam {self.camera_id} | "
                    f"count={self.counter_fire_frame} frame "
                    f"avg_fire_conf={avg_fire_conf:.2f} "
                    f"max_conf={self.max_fire_conf:.2f} "
                )
                if self.show_video:
                    cv2.putText(
                        annotated_frame,
                        warning_text,
                        (10, int(h*0.2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA
                    )
        # ====== Mess Smoke LERT ======
        if self.counter_smoke_frame > self.smoke_warring_time:
            avg_smoke_conf = np.mean(self.smoke_confs) if self.smoke_confs else 0
            if self.show_video:
                cv2.putText(
                    annotated_frame,
                    f"Warring:  Smoke Certainty {min(100,(avg_smoke_conf *100) + (self.counter_smoke_frame - self.smoke_warring_time)*0.2):.0f}%",
                    (10,int(h*0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
            if (
                self.counter_smoke_frame >= 20 and
                avg_smoke_conf >= 0.5 and
                self.max_smoke_conf >= 0.6 and
                not self.smoke_alerted
            ):
                self.smoke_alerted = True
                if self.send_api:
                    send_to_kafka(self.box_id, self.cam_id, "smoke", self.max_smoke_conf)
                # start record
                self.start_recording("smoke")
                # snapshot ngay
                self.save_snapshot(annotated_frame, "smoke", "now")
                # reset timer snapshot 10s
                self.smoke_alert_time = time.time()
                logging.warning(
                    f"🔥 SMOKE ALERT Cam {self.camera_id} | "
                    f"count={self.counter_smoke_frame} frame "
                    f"avg_smoke_conf={avg_smoke_conf:.2f} "
                    f"max_conf={self.max_smoke_conf:.2f} "
                )
                if self.show_video:
                    cv2.putText(
                        annotated_frame,
                        warning_text,
                        (10, int(h*0.2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA
                    )
        return annotated_frame

    def reset_fire_state(self):
        self.counter_fire_frame = 0
        self.reset_fire_frame = 0
        self.fire_confs.clear()
        self.max_fire_conf = 0.0
        self.fire_alerted = False
    def reset_smoke_state(self):
        self.counter_smoke_frame = 0
        self.reset_smoke_frame = 0
        self.smoke_confs.clear()
        self.max_smoke_conf = 0.0
        self.smoke_alerted = False

    def run(self):
        try:
            while True:
                if self.is_video_file:
                    ret, frame = self.cap.read()
                else:
                    ret, frame = self.rtsp_stream.get_frame()
                if not ret or frame is None:
                    if self.is_video_file:
                        break
                    time.sleep(0.2)
                    continue
                
                annotated = self.process_frame(frame)
                self.frame_buffer.append(annotated.copy())
                if self.recording:
                    self.record_writer.write(cv2.resize(annotated, (self.width, self.height)))

                    elapsed = time.time() - self.record_start_time
                    if elapsed >= self.record_seconds:
                        self.stop_recording()
                if self.fire_alerted and not self.snapshot_10s_taken:
                    if time.time() - self.fire_alert_time >= 10:
                        self.save_snapshot(annotated, "fire", "after10s")
                        self.snapshot_10s_taken = True
                if self.smoke_alerted and not self.snapshot_10s_taken:
                    if time.time() - self.smoke_alert_time >= 10:
                        self.save_snapshot(annotated, "smoke", "after10s")
                        self.snapshot_10s_taken = True
                if self.show_video:
                    self.video_writer.write(cv2.resize(annotated, (self.width, self.height)))
                    cv2.imshow(f"Cam {self.camera_id}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.cleanup()

    def cleanup(self):
        if hasattr(self, 'video_writer') and self.video_writer:
            self.video_writer.release()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'rtsp_stream'):
            self.rtsp_stream.stop()
        cv2.destroyAllWindows()


# ====================== THREAD ======================
def run_tracker_for_camera(store_id, camera_id, output_path, size, show_video, send_api, video_path=None):
    tracker = None
    try:
        tracker = CustomerTracker(store_id, camera_id, output_path, size, show_video, send_api, video_path=video_path)
        tracker.run()
    except Exception as e:
        logging.error(f"Camera {camera_id} error: {e}")
    finally:
        if tracker:
            tracker.cleanup()

# ====================== MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_id', type=int, nargs='*', default=None, help="ID của camera cần chạy")
    parser.add_argument('--store_id', type=str, default='vn316', help="Mã cửa hàng")
    parser.add_argument('--output', type=str, default='./output/video.mp4', help="path output video")
    parser.add_argument('--imgsz', type=int, default=480, help="size input images")
    parser.add_argument('--show_video', action='store_true', help="show results video")
    parser.add_argument('--send_api', action='store_true', help="Send data to kafka")
    parser.add_argument('--video_path', type=str, default=r"C:\Users\OS\Desktop\firesmoke\video_fire1.mp4", help="Đường dẫn file video để chạy thay vì camera")

    args = parser.parse_args()

    # Nếu có video_path -> chỉ chạy 1 luồng video
    if args.video_path:
        t = threading.Thread(
            target=run_tracker_for_camera,
            args=(args.store_id, 0, args.output, args.imgsz, args.show_video, args.send_api, args.video_path)
        )
        t.start()
        t.join()
    else:
        cam_ids = args.camera_id or list(camera_configs[args.store_id]["cameras"].keys())
        threads = []
        for cid in cam_ids:
            t = threading.Thread(
                target=run_tracker_for_camera,
                args=(args.store_id, cid, args.output, args.imgsz, args.show_video, args.send_api, None)
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
