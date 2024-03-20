# Use an appropriate base image
FROM tensorflow/tensorflow:2.7.0-gpu

# copy the requirements.txt file
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace/shiva_wmh

ARG UID
ARG GID
ARG USER

RUN groupadd -g $GID -o $USER
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $USER
USER $USER

# Expose any necessary ports
EXPOSE 8080

# Specify the command to run when the container starts
CMD ["bash"]

