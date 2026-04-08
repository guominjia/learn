## Uploading the distribution archives

The first thing you’ll need to do is register an account on TestPyPI, which is a separate instance of the package index intended for testing and experimentation.
It’s great for things like this tutorial where we don’t necessarily want to upload to the real index.
To register an account, go to https://test.pypi.org/account/register/ and complete the steps on that page. You will also need to verify your email address before you’re able to upload any packages.
For more details, see [Using TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/).

To securely upload your project, you’ll need a PyPI [API token](https://test.pypi.org/help/#apitoken). Create one at https://test.pypi.org/manage/account/#api-tokens, setting the “Scope” to “Entire account”. Don’t close the page until you have copied and saved the token — you won’t see that token again.

Now that you are registered, you can use [twine](https://packaging.python.org/en/latest/key_projects/#twine) to upload the distribution packages. You’ll need to install Twine:

```
pip install --upgrade twine
```

Once installed, run Twine to upload all of the archives under dist:

```
python -m twine upload --repository testpypi dist/*
```

You will be prompted for an API token. Use the token value, including the `pypi-` prefix. Note that the input will be hidden, so be sure to paste correctly.

After the command completes, you should see output similar to this:

```
Uploading distributions to https://test.pypi.org/legacy/
Enter your API token:
Uploading example_package_YOUR_USERNAME_HERE-0.0.1-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.2/8.2 kB • 00:01 • ?
Uploading example_package_YOUR_USERNAME_HERE-0.0.1.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.8/6.8 kB • 00:00 • ?
```
Once uploaded, your package should be viewable on TestPyPI; for example: `https://test.pypi.org/project/example_package_YOUR_USERNAME_HERE`

## References
- <https://packaging.python.org/en/latest/tutorials/packaging-projects>