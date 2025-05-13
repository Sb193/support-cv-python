FROM python:3.9-slim

WORKDIR /app

# Cài đặt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cấu hình pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Cài đặt Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn
COPY . .

# Mở port
EXPOSE 8080

# Chạy ứng dụng
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
