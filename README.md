# Project overview

This is the beginning of `q-gen`, an event generator for QuIPS. The current layout of the project is given in `DESIGN.md`. This is just one view of what it could look like! Soon, there will be a link that you can click that gives a simple interactive demonstration of the output

Add any features you think are interesting in a new branch, and we can merge into main once they work.

A small note: The project is currently written to run python via the uv command [(see here to install)](https://docs.astral.sh/uv/), as it's nice for handling package dependencies. If you have a preferred method of doing this, feel free to amend this.

## Using browser visualization

To pull up the browser interface, `git clone` this directory. From inside the project directory, you first `uv sync' to make sure that all the required python packages are installed, then `uv run uvicorn viz.server:app --reload --port 8000`. Then open `http://localhost:8000/` in your browser.
