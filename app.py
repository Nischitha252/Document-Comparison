"""
Entry point for the RFQ Cybersecurity Streamlit application.

This module stays deliberately lightweight, delegating all heavy
lifting to the ``rfq_cybersecurity`` package.  Keeping this file
concise improves readability and makes it straightforward to
start the app locally or deploy it into a production environment.

To launch the app, run ``streamlit run app.py`` from the project
root.  The :func:`main` function simply calls :func:`run_app`
from :mod:`rfq_cybersecurity.ui.main_view`.
"""

import streamlit as st  # type: ignore

# Import the top-level UI function from the package
from rfq_cybersecurity.ui import run_app 


def main() -> None:
    """Run the Streamlit application.

    This function delegates to :func:`run_app` in the
    :mod:`rfq_cybersecurity.ui` package.  Defining a separate
    ``main`` function makes it easier to hook into other
    frameworks or testing harnesses in the future.
    """
    run_app()


if __name__ == "__main__":
    # When executed as a script, run the application.
    main()