FROM ghost-live-cloud-serverless

COPY . /PyAV-wrapper/

RUN pip install --force-reinstall --no-deps /PyAV-wrapper/


CMD ["timeout","60","python3", "-u", "listen.py"]


