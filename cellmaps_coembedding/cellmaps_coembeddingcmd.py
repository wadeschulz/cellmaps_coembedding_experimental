#! /usr/bin/env python

import argparse
import os
import sys
import logging
import logging.config

from cellmaps_coembedding.exceptions import CellmapsCoEmbeddingError
from cellmaps_utils import logutils
from cellmaps_utils import constants
import cellmaps_coembedding
from cellmaps_coembedding.runner import MuseCoEmbeddingGenerator
from cellmaps_coembedding.runner import FakeCoEmbeddingGenerator
from cellmaps_coembedding.runner import CellmapsCoEmbedder

logger = logging.getLogger(__name__)

PPI_EMBEDDINGDIR = '--ppi_embeddingdir'
IMAGE_EMBEDDINGDIR = '--image_embeddingdir'


def _parse_arguments(desc, args):
    """
    Parses command line arguments

    :param desc: description to display on command line
    :type desc: str
    :param args: command line arguments usually :py:func:`sys.argv[1:]`
    :type args: list
    :return: arguments parsed by :py:mod:`argparse`
    :rtype: :py:class:`argparse.Namespace`
    """
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=constants.ArgParseFormatter)
    parser.add_argument('outdir', help='Output directory')
    parser.add_argument('--embeddings', nargs='+',
                        help='Directories with embeddings. Requires two or more paths.')
    parser.add_argument('--algorithm', choices=['auto', 'muse'], default='auto',
                        help='Algorithm to use for coembedding. Defaults to auto.')
    parser.add_argument(PPI_EMBEDDINGDIR,
                        help='Directory aka rocrate where ppi '
                             'embedding file resides (Deprecated: use embeddings flag)')
    parser.add_argument(IMAGE_EMBEDDINGDIR,
                        help='Directory aka rocrate image embedding '
                             'file resides (Deprecated: use embeddings flag)')
    parser.add_argument('--latent_dimension', type=int, default=128,
                        help='Output dimension of embedding')
    parser.add_argument('--n_epochs_init', default=200, type=int,
                        help='# of init training epochs')
    parser.add_argument('--n_epochs', default=500, type=int,
                        help='# of training epochs')
    parser.add_argument('--jackknife_percent', default=0.0, type=float,
                        help='Percentage of data to withhold from training'
                             'a value of 0.1 means to withhold 10 percent of the data')
    parser.add_argument('--fake_embedding', action='store_true',
                        help='If set, generate fake coembeddings')
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--skip_logging', action='store_true',
                        help='If set, output.log, error.log '
                             'files will not be created')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help='Increases verbosity of logger to standard '
                             'error for log messages in this module. Messages are '
                             'output at these python logging levels '
                             '-v = WARNING, -vv = INFO, '
                             '-vvv = DEBUG, -vvvv = NOTSET (default ERROR '
                             'logging)')
    parser.add_argument('--version', action='version',
                        version=('%(prog)s ' +
                                 cellmaps_coembedding.__version__))

    return parser.parse_args(args)


def set_embedding_dirs(embeddings):
    dirs = []
    file_names = []

    for embed in embeddings:
        if os.path.isfile(embed):
            dirs.append(os.path.dirname(embed))
            file_names.append(os.path.basename(embed))
        else:
            dirs.append(embed)
            file_names.append(None)

    return dirs, file_names if file_names else None


def main(args):
    """
    Main entry point for program

    :param args: arguments passed to command line usually :py:func:`sys.argv[1:]`
    :type args: list

    :return: return value of :py:meth:`cellmaps_coembedding.runner.CellmapsCoEmbedder.run`
             or ``2`` if an exception is raised
    :rtype: int
    """
    desc = """
    Version {version}

    Given image and PPI embeddings, this tool generates a co-embedding using
    a variant of MuSE algorithm within this code base from
    Feng Bao @ Altschuler & Wu Lab @ UCSF 2022
    that is under MIT License.

    To run this tool requires that an output directory be specified and two embeddings
    be set via --embeddings flag. The values passed to --embeddings can be an ro-crate 
    containing either a ppi_emd.tsv or image_emd.tsv file or a path to a TSV file.
    
    It is assumed these files are tab delimited embeddings and for each row, 
    first value is assumed to be sample ID followed by the embeddings separated by 
    tabs. The first row is assumed to be a header. 

    

    """.format(version=cellmaps_coembedding.__version__)
    theargs = _parse_arguments(desc, args[1:])
    theargs.program = args[0]
    theargs.version = cellmaps_coembedding.__version__

    if (theargs.ppi_embeddingdir or theargs.image_embeddingdir) and theargs.embeddings:
        raise CellmapsCoEmbeddingError('Use either --ppi_embeddingdir and --image_embeddingdir or --embeddings, '
                                       'not both')
    embed_files = None
    if theargs.embeddings:
        if len(theargs.embeddings) == 2 and (theargs.algorithm == 'auto' or theargs.algorithm == 'muse'):
            embed_dirs, embed_files = set_embedding_dirs(theargs.embeddings)
            theargs.ppi_embeddingdir = embed_dirs[0]
            theargs.image_embeddingdir = embed_dirs[1]
        else:
            raise CellmapsCoEmbeddingError('Currently, only two directories are supported with --embeddings')

    if theargs.ppi_embeddingdir and theargs.image_embeddingdir:
        input_dirs = [theargs.image_embeddingdir, theargs.ppi_embeddingdir]
    else:
        raise CellmapsCoEmbeddingError('Either --ppi_embeddingdir and --image_embeddingdir, '
                                       'or --embeddings are required')

    try:
        logutils.setup_cmd_logging(theargs)
        if theargs.fake_embedding:
            gen = FakeCoEmbeddingGenerator(dimensions=theargs.latent_dimension,
                                           ppi_embeddingdir=theargs.ppi_embeddingdir,
                                           image_embeddingdir=theargs.image_embeddingdir,
                                           embedding_filenames=embed_files)
        else:
            gen = MuseCoEmbeddingGenerator(dimensions=theargs.latent_dimension,
                                           n_epochs=theargs.n_epochs,
                                           n_epochs_init=theargs.n_epochs_init,
                                           jackknife_percent=theargs.jackknife_percent,
                                           outdir=os.path.abspath(theargs.outdir),
                                           ppi_embeddingdir=theargs.ppi_embeddingdir,
                                           image_embeddingdir=theargs.image_embeddingdir,
                                           embedding_filenames=embed_files)
        return CellmapsCoEmbedder(outdir=theargs.outdir,
                                  inputdirs=input_dirs,
                                  embedding_generator=gen,
                                  skip_logging=theargs.skip_logging,
                                  input_data_dict=theargs.__dict__).run()
    except Exception as e:
        logger.exception('Caught exception: ' + str(e))
        return 2
    finally:
        logging.shutdown()


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))
