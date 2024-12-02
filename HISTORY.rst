=======
History
=======

0.4.0 (2024-12-02)
-------------------

* Added README generation.

* Refactor code.

0.3.1 (2024-09-13)
------------------

* Bug fix: raise more informative error when no embeddings overlap.

0.3.0 (2024-09-06)
------------------

* Added ``--provenance`` flag to pass a path to json file with provenance information. This removes the
  necessity of input directory to be an RO-Crate.

0.2.0 (2024-07-17)
------------------

* Added a new coembedding algorithm accessible via flag ``--algorithm auto``. This algorithm utilizes neural networks
  to generate latent embeddings, optimizing both reconstruction and triplet losses to improve embedding accuracy
  by learning intra- and inter-modality relationships.

0.1.0 (2024-02-12)
------------------

* First release on PyPI.
