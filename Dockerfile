FROM ollama/ollama

WORKDIR /root

COPY requirements.txt ./

RUN apt update \
    && apt-get install -y python3 python3-pip python3-dev build-essential vim git \
    && pip install --break-system-packages -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8501
EXPOSE 11434
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

