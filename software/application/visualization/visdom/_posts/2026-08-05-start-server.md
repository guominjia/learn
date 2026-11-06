---
layout: post
title: "Start a Visdom Server with Python"
date: 2026-08-05
categories: [visualization, python]
tags: [visdom, python, machine-learning, dashboard]
---

Visdom is a browser-based dashboard for visualizations sent from Python programs. A training script can publish plots, images, and text to a Visdom server while it runs, making it useful for inspecting experiments without adding plotting code to every iteration.

The server is included with the Python package. The most explicit way to start it is to run its server module:

```powershell
python -m visdom.server
```

This command starts the same server as the `visdom` command-line entry point. By default, open the dashboard in a browser at:

```text
http://localhost:8097
```

Keep the terminal running while the dashboard is in use. Press `Ctrl+C` in that terminal to stop the server.

## Install Visdom

Install the package into the same Python environment that will run the server and the client code:

```powershell
python -m pip install visdom
```

Using `python -m pip` and `python -m visdom.server` is helpful when multiple Python installations or virtual environments are available. Both commands then use the interpreter selected by `python`.

To confirm that the module is available before opening the dashboard, run:

```powershell
python -m visdom.server --help
```

If Python cannot find the `visdom` module, activate the intended virtual environment and install the package again in that environment.

## Send Data from Python

Once the server is running, a client can connect with the default `Visdom` constructor. The following example adds a text pane and a line plot to the `training` environment:

```python
import visdom

vis = visdom.Visdom(env="training")

if not vis.check_connection():
	raise RuntimeError("Could not connect to the Visdom server")

vis.text("The Visdom client is connected.")
vis.line(
	X=[1, 2, 3, 4],
	Y=[0.91, 0.72, 0.58, 0.47],
	opts={"title": "Training loss", "xlabel": "epoch", "ylabel": "loss"},
)
```

The default client configuration targets `http://localhost` on port `8097`, so no connection parameters are needed when the client and server run on the same machine. Refresh the dashboard after running the script to see the new windows.

## Use Another Port

When port `8097` is already occupied, start Visdom on a different port:

```powershell
python -m visdom.server -port 8098
```

The client must use the same port:

```python
vis = visdom.Visdom(port=8098, env="training")
```

Open `http://localhost:8098` in the browser in this case.

## Access a Remote Server

For a server running on another machine, configure the client with that server's URL and port:

```python
vis = visdom.Visdom(server="http://ml-host.example", port=8097, env="training")
```

Do not expose an unauthenticated development dashboard directly to an untrusted network. Visdom supports server options for login and for binding only to localhost; use the upstream command-line documentation when configuring remote access.

## Quick Checklist

1. Install the package with `python -m pip install visdom`.
2. Start the server with `python -m visdom.server`.
3. Open `http://localhost:8097`.
4. Run a Python client that creates a `visdom.Visdom()` instance.
5. Use `vis.check_connection()` to fail early when the client cannot reach the server.

## References

- [Visdom README](https://github.com/fossasia/visdom): documents installation, the equivalence of `visdom` and `python -m visdom.server`, the default `http://localhost:8097` endpoint, client defaults, and server command-line options.
