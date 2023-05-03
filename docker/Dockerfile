# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

RUN apt-get --allow-releaseinfo-change update
RUN apt-get install -y build-essential 

RUN mkdir /tmp/cellmaps_coembedding
COPY ./ /tmp/cellmaps_coembedding/
RUN pip install /tmp/cellmaps_coembedding

RUN rm -rf /tmp/cellmaps_coembedding

ENTRYPOINT ["/opt/conda/bin/cellmaps_coembeddingcmd.py"]

CMD ["--help"]
