from datetime import datetime
import json
import pytz
from confluent_kafka import Producer
import logging
logging.basicConfig(
    level=logging.INFO,  # hoặc DEBUG nếu bạn muốn xem chi tiết
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Biến toàn cục để lưu trữ instance Producer
_producer = None
_producer_config = {
    'bootstrap.servers': 'hcm.gateway01.cxview.ai:9093',
    'client.id': 'python-producer',
    'security.protocol': 'PLAINTEXT',
    # 'security.protocol': 'SSL',
    # 'ssl.ca.location': './cert/ca-root.pem',
    # 'ssl.certificate.location': './cert/ca-cert.pem',
    # 'ssl.key.location': './cert/ca-key.pem',
    'retries': 5,  # Thêm retry để tăng độ tin cậy
    'retry.backoff.ms': 1000,  # Chờ 1 giây giữa các lần thử
    'debug': 'security'
}
def rescale(frame, img_size, x_min, y_min, x_max, y_max):
    scale_x = frame.shape[1] / img_size
    scale_y = frame.shape[0] / img_size
    return int(x_min * scale_x), int(y_min * scale_y), int(x_max * scale_x), int(y_max * scale_y)

def get_producer():
    """Lấy hoặc khởi tạo Producer một lần duy nhất."""
    global _producer
    if _producer is None:
        try:
            _producer = Producer(_producer_config)
            logging.info("Kafka Producer initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Kafka Producer: {str(e)}")
            raise
    return _producer

def send_to_kafka(box_id, cam_id, type, accuracy, topic="production-gatewayhn-peoplecount"):
    """Gửi thông điệp đến Kafka topic, sử dụng Producer chung."""
    timestamp = int(datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).timestamp())
    # Chuẩn bị dữ liệu theo metric
    data = {
        "box_id": box_id,
        "cam_id":cam_id,
        "type": type,
        "accuracy": accuracy,
        "timestamp": timestamp
    }

    def delivery_report(err, msg):
        if err is not None:
            logging.error(f"Error sending message: {err}")
        else:
            logging.info(
                data
            )
    try:
        producer = get_producer()
        message = json.dumps(data)
        producer.produce(topic, value=message.encode('utf-8'), callback=delivery_report)
        producer.poll(5)
    except Exception as e:
        logging.error(f"Error sending message to Kafka: {str(e)}")
        raise


def close_producer():
    """Đóng Producer khi ứng dụng kết thúc."""
    global _producer
    if _producer is not None:
        try:
            _producer.flush()  # Đảm bảo tất cả thông điệp được gửi trước khi đóng
            logging.info("Kafka Producer closed successfully")
        except Exception as e:
            logging.error(f"Error closing Kafka Producer: {str(e)}")
        finally:
            _producer = None


