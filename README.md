Metric analysis scripts
=======================

Supported competitions:
 * DSB 2018 (Kaggge DSB format)
 * Sartorius (Kaggge DSB format)
 * MIDOG 2021 (Custom format)

First, the stardist `master` of version `f73cdc4` should be installed locally from github (with `pip -e`, for example, then the patch [stardist_matching.patch](stardist_matching.patch) should be applied

DSB 2018
--------

One needs a gold CSV and the submissions in zip format (for each submittion, the zip file contains the submission CSV).

```bash
    python3 evaluate_dsb_like.py \
        --submissions-path ... \
        --gold-path ... \
        --output-path ... \
        --evaluator=dsb2018 \
        --by-image
```

Sartorius
---------

The arguments are the same, except the evaluator should be `sartorius`.

MIDOG 2021
-----

```bash
    python3 evaluate_midog_custom.py 
        --submissions-path ... \
        --gold-path ... \
        --output-path midog_result
        --matching-path ... \
        --pixel-sizes ... \
```

One needs two additional files: the matching CSV and the pixel density for each image.

The results for each team will be saved into the output folder.

It can be summarized using the `summary.py` or any other script.