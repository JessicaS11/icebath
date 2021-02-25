icebath
=======
Tools for inferring bathymetry using icebergs as depth sounders

 |GitHub license|

.. |GitHub license| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

Licensing
---------

The content of this project is licensed under the `Creative Commons Attribution 3.0 Unported license <https://creativecommons.org/licenses/by/3.0/>`_, and the underlying source code used to format and display that content is licensed under the `BSD-3-Clause <LICENSE.rst>`_.

Contributing
------------
Find something useful? Modify or generalize one of the functions for your own use? I'd love for you to contribute your changes and improvements back to icebath. Feel free to submit a PR or send me a message about adding your code or example workflow!

Notes
-----
This code has some important things to be aware of r.e. geospatial handling (coordinate reference systems (CRS), projections, etc.). Many, but not all, of this instances are printed out when running the code. The reason for this is code development began using Xarray and some rasterio. Rioxarray (the Xarray-rasterio engine/wrapper) was introduced partway through development, but older code was not fully converted (some of the changes were breaking to steps later in the workflow). Thus, the workflow has some combination of automated and "manual" or customized handling of geospatial concerns (CRS, transforms, projections, array orientation given negative dimension/coordinate handling in Xarray), and often is not explicitly checked or reported to the user. If you run into any empty array/no result/wonky spacing errors (where you otherwise know you should be getting some type of reasonable results), there's a good chance that a dataset being flipped somewhere within your workflow is the culprit. Hopefully someday I'll be able to convert the entire workflow to Xarray with a rioxarray engine and do a more robust job of handling geospatial parameters. PRs for that or adding other functionality are always welcome.

Similarly, the use of dask for parallelization was introduced partway through development. Thus, not all functions have been set up to [effectively] use dask [well] (this is especially true in `icebath.core.build_gdf()`). I hope to keep working on this as time allows.
