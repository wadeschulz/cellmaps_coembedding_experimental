=====
Usage
=====

This page should provide information on how to use cellmaps_coembedding

In a project
--------------

To use cellmaps_coembedding in a project::

    import cellmaps_coembedding
    
    
Needed files
------------

The output directories for the image embeddings (see `Cell Maps Image Embedding <https://github.com/idekerlab/cellmaps_image_embedding/>`__) and protein-protein interaction network embeddings (see `Cell Maps PPI Embedding <https://github.com/idekerlab/cellmaps_ppi_embedding/>`__) are required. 


On the command line
---------------------

For information invoke :code:`cellmaps_coembeddingcmd.py -h`

**Example usage**

.. code-block::

   cellmaps_coembeddingcmd.py ./cellmaps_coembedding_outdir --image_embeddingdir ./cellmaps_image_embedding_outdir --ppi_embeddingdir ./cellmaps_ppi_embedding_outdir 

Via Docker
---------------

**Example usage**


.. code-block::

   docker run -v `pwd`:`pwd` -w `pwd` idekerlab/cellmaps_coembedding:0.1.0 cellmaps_coembeddingcmd.py ./cellmaps_coembedding_outdir --image_embeddingdir ./cellmaps_image_embedding_outdir --ppi_embeddingdir ./cellmaps_ppi_embedding_outdir 



