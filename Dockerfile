# Sử dụng base image có CUDA Toolkit
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$CUDA_PATH/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    librdkafka-dev \
    libcudnn8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Xoá source để giảm dung lượng image
RUN rm -rf /opt/opencv /opt/opencv_contrib
# Cài numpy trước để tránh lỗi khi install deep-person-reid
RUN pip3 install numpy

# Cài requirements chung trước (không chứa deep-person-reid)
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Sao chép mã nguồn
WORKDIR /app
COPY . .

# Sao chép file run_schedule.sh và cấp quyền thực thi
COPY run_schedule.sh /run_schedule.sh
RUN chmod +x /run_schedule.sh
RUN apt-get update && apt-get install -y tzdata
# Thiết lập múi giờ Việt Nam
ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Thiết lập entrypoint để chạy script kiểm tra thời gian
ENTRYPOINT ["/run_schedule.sh"]
