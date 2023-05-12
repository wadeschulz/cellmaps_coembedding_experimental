#! /usr/bin/env python

import argparse
import sys
import logging
import logging.config

from cellmaps_utils import logutils
from cellmaps_utils import constants
import cellmaps_coembedding
from cellmaps_coembedding.runner import FakeCoEmbeddingGenerator
from cellmaps_coembedding.runner import CellmapsCoEmbedder

logger = logging.getLogger(__name__)


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
    parser.add_argument('--ppi_embeddingdir', required=True,
                        help='Directory aka rocrate where ppi '
                             'embedding file resides')
    parser.add_argument('--image_embeddingdir', required=True,
                        help='Directory aka rocrate image embedding '
                             'file resides')
    parser.add_argument('--image_downloaddir', required=True,
                        help='Directory containing image download data or'
                             'more specifically rocrate where '
                             'file resides')
    parser.add_argument('--latent_dimension', type=int, default=128,
                        help='Output dimension of embedding')
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increases verbosity of logger to standard '
                             'error for log messages in this module. Messages are '
                             'output at these python logging levels '
                             '-v = ERROR, -vv = WARNING, -vvv = INFO, '
                             '-vvvv = DEBUG, -vvvvv = NOTSET (default no '
                             'logging)')
    parser.add_argument('--version', action='version',
                        version=('%(prog)s ' +
                                 cellmaps_coembedding.__version__))

    return parser.parse_args(args)


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

    Invokes run() method on CellmapsCoEmbedder

    """.format(version=cellmaps_coembedding.__version__)
    theargs = _parse_arguments(desc, args[1:])
    theargs.program = args[0]
    theargs.version = cellmaps_coembedding.__version__

    try:
        logutils.setup_cmd_logging(theargs)
        gen = FakeCoEmbeddingGenerator(dimensions=theargs.latent_dimension,
                                       ppi_embeddingdir=theargs.ppi_embeddingdir,
                                       image_embeddingdir=theargs.image_embeddingdir,
                                       image_downloaddir=theargs.image_downloaddir)
        return CellmapsCoEmbedder(outdir=theargs.outdir,
                                  inputdirs=[theargs.image_embeddingdir, theargs.ppi_embeddingdir,
                                             theargs.image_downloaddir],
                                  embedding_generator=gen,
                                  input_data_dict=theargs.__dict__).run()
    except Exception as e:
        logger.exception('Caught exception: ' + str(e))
        return 2
    finally:
        logging.shutdown()


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))
