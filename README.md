<div align=center>
  <h1>
  :pencil2: Sequential Sketch Stroke Generation
  </h1>
  <p>
    <a href=https://mhsung.github.io/kaist-cs492d-fall-2024/ target="_blank"><b>KAIST CS492(D): Diffusion Models and Their Applications (Fall 2024)</b></a><br>
    Course Project
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://mhsung.github.io target="_blank"><b>Minhyuk Sung</b></a> (mhsung [at] kaist.ac.kr)<br>
    TA: <a href=https://phillipinseoul.github.io/ target="_blank"><b>Yuseung Lee</b></a>  (phillip0701 [at] kaist.ac.kr)
  </p>
</div>

<div align=center>
   <img src="./assets/teaser.png">
   <figcaption>
    Collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw!. Drawings were captured as timestamped vectors.
    <i>Source: <a href="https://quickdraw.withgoogle.com/data/">Quick, Draw! Dataset.</a></i>
    </figcaption>
</div>

## Description
In this project, your goal is to implement a conditional diffusion model that generates sequential strokes to form a sketch. You will utilize user-captured sketch strokes, complete with timestamps, from 345 different object categories provided by the [Quick, Draw!](https://quickdraw.withgoogle.com/data/) dataset. Rather than generating the entire sketch at once, the focus should be on leveraging the sequential (or part-aware) stroke information for training the model, encouraging a stroke-by-stroke generation process that reflects how users naturally draw.

## Data Specification
The dataset consists of sketches from 345 different categories, and each sketch is drawn with varying number of strokes.

Use the following bash script to download the Quick, Draw! dataset:
```
sh download_quickdraw.sh
```

For extracting the `.ndjson` file for each category and visualizing the source data, refer to our sample code in `load_data.ipynb`. An example sequence of strokes is shown below:

<div align=center>
  <img src="./assets/sample.png" width="768"/>
</div>

## Tasks
Your task is to implement a conditional diffusion model that generates sequential strokes to form a sketch. Report evaluation metrics using the test dataset.

## Evaluation
We will evaluate your model on the test dataset. Only use the training dataset to train your model. 

## Acknowledgement 
We appreciate Google Creative Lab for releasing the [Quick, Draw!](https://oppo-us-research.github.io/OpenIllumination/) dataset to public. The contents adhere to the [Creative Commons License](https://creativecommons.org/licenses/by/4.0/).