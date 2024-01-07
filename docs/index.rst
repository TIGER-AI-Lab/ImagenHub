.. ImagenHub documentation master file, created by
   sphinx-quickstart on Tue Oct 10 17:07:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ImagenHub's documentation!
=====================================

ImagenHub is a centralized framework to standardize the evaluation of conditional image generation models by curating unified datasets, building an inference library and a benchmark that align with real life applications.

.. _Project Page: https://tiger-ai-lab.github.io/ImagenHub/
.. _Arxiv Paper: https://arxiv.org/abs/2310.01596
.. _HuggingFace Datasets: https://huggingface.co/ImagenHub
.. _GitHub Code: https://github.com/TIGER-AI-Lab/ImagenHub
.. _Imagen Museum: https://chromaica.github.io/#imagen-museum

* `Project Page`_
* `Arxiv Paper`_
* `HuggingFace Datasets`_
* `GitHub Code`_
* `Imagen Museum`_

.. toctree::
   :maxdepth: 2
   :caption: Overview

   Overview/intro
   Overview/philosophy
   Overview/models
   Overview/datasets

.. toctree::
   :maxdepth: 2
   :caption: Guidelines

   Guidelines/install
   Guidelines/quickstart
   Guidelines/custombenchmark
   Guidelines/deepdive

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Researcher Guidelines

   Guidelines/custommodel
   Guidelines/humaneval
   Guidelines/imagenmuseum

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contributing

   Contributing/basics
   Contributing/coding
   Contributing/docs

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   source/modules
   source/imagen_hub.infermodels
   source/imagen_hub.loader
   source/imagen_hub.benchmark
   source/imagen_hub.metrics
   source/imagen_hub.miscmodels
   source/imagen_hub.utils
   source/imagen_hub.depend
   source/imagen_hub.pipelines
