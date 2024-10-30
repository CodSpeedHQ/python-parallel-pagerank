# Pagerank performance measurement with python 3.13

Running:

- Without GIL:

  ```
  uv run python -X gil=0 -m pytest --codspeed -vs -x --codspeed-max-time 10 tests.py
  ```

- With GIL

  ```
  uv run python -X gil=1 -m pytest --codspeed -vs -x --codspeed-max-time 10 tests.py
  ```
