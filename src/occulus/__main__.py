"""Entry point for ``python -m occulus``.

Delegates to the CLI main function so the package can be invoked
directly from the command line::

    python -m occulus info scan.laz
    python -m occulus classify scan.laz -o ground.laz --algorithm csf
"""

import sys

from occulus.cli.main import main

sys.exit(main())
