# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

RUN apt-get --allow-releaseinfo-change update
RUN apt-get install -y build-essential 

RUN mkdir /tmp/cellmaps_generate_ppi
COPY ./ /tmp/cellmaps_generate_ppi/
RUN pip install /tmp/cellmaps_generate_ppi

RUN rm -rf /tmp/cellmaps_generate_ppi

ENTRYPOINT ["/opt/conda/bin/cellmaps_generate_ppicmd.py"]

CMD ["--help"]
