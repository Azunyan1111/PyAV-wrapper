FROM ghost-live-cloud-serverless-azunyan1111

COPY . /PyAV-wrapper/

RUN pip install --force-reinstall --no-deps /PyAV-wrapper/

ENV THREAD_SIZE=5
ENV BATCH_SIZE=5

CMD ["timeout","60","python3", "-u", "listen.py"]
