# ConvNet - Style Transfer
_Using PyTorch to apply artistic styles to pictures_

---

## About

This is my first ML-Project using the PyTorch library. Furthermore, I want to gain some experience with advanced
ConvNet technologies and wanted to have some fun with the creative parts of machine learning.

## Usage

The style-transfer script is built to work like any other command-line application. You call it using python and pass some of the following arguments to it:

| Argument          | Explanation                                                       | Required? | Default             |
|-------------------|-------------------------------------------------------------------|-----------|---------------------|
| -h, --help        | Displays the help message                                         | - [ ]     | -                   |
|                   |                                                                   |           |                     |
| -c, --content     | Specifies the location of the content-image                       | - [x]     | -                   |
| -s, --style       | Specifies the location of the style-image                         | - [x]     | -                   |
| -d, --destination | Specifies the destination of the resulting image                  | - [ ]     | -                   |
|                   |                                                                   |           |                     |
| -v, --verbose     | Flag; Should intermediate results be displayed?                   | - [ ]     | False               |
| -i, --interval    | Defines the interval in which current results should be displayed | - [ ]     | 10                  |
|                   |                                                                   |           |                     |
| --lr              | Defines the learning rate                                         | - [ ]     | 0.1                 |
| --content-w       | Provides the content weight                                       | - [ ]     | 1                   |
| --style-w         | Provides the style weight                                         | - [ ]     | 10<sup>5</sup>      |
| --tv-w            | Provides the total variation weight                               | - [ ]     | 3 * 10<sup>-3</sup> |

## Results

After a couple of bug fixes, here are some of the final results:

| Content                                                 | Style                                                                | Result                                                         |
|---------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------|
| ![man-beard](media/content/man-beard-scaled.jpg)        | ![starry-night](media/style/van-gogh-starry-night-scaled.jpg)        | ![result](media/results/man-beard_starry-night.jpg)            |
| ![ballerina](media/content/ballerina-scaled-scaled.jpg) | ![tarantula-nebula](media/style/tarantula-nebula-scaled.jpg)         | ![result](media/results/ballerina_tarantula-nebula_scaled.jpg) |
| ![lake-pier](media/content/lake-pier-scaled-scaled.jpg) | ![toyokawa-bridge](media/style/hiroshige-toyokawa-bridge-scaled.jpg) | ![result](media/results/lake-pier_toyokawa-bridge_scaled.jpg)  |

_Various examples_

| ...                                                      | August Macke - "Four Girls"                                   | Gustav Klimt - "The Kiss"                         | Hiroshige - "Toyokawa Bridge"                                        | The Tarantula Nebula                                         | Vincent van Gogh - "Starry Night"                             | Edvard Munch - "The Scream"                           |
|----------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------|---------------------------------------------------------------|-------------------------------------------------------|
|                                                          | ![four-girls](media/style/august-macke-four-girls-scaled.jpg) | ![kiss](media/style/gustav-klimt-kiss-scaled.jpg) | ![toyokawa-bridge](media/style/hiroshige-toyokawa-bridge-scaled.jpg) | ![tarantula-nebula](media/style/tarantula-nebula-scaled.jpg) | ![starry-night](media/style/van-gogh-starry-night-scaled.jpg) | ![scream](media/style/edvard-munch-scream-scaled.jpg) |
| ![woman-smiling](media/content/woman-smiling-scaled.jpg) | ![result](media/results/woman-smiling_four-girls.jpg)         | ![result](media/results/woman-smiling_kiss.jpg)   | ![result](media/results/woman-smiling_toyokawa-bridge.jpg)           | ![result](media/results/woman-smiling_tarantula-nebula.jpg)  | ![result](media/results/woman-smiling_starry-night.jpg)       | ![result](media/results/woman-smiling_scream.jpg)     |
| ![parrots-kiss](media/content/parrots-kiss-scaled.jpg)   | ![result](media/results/parrots-kiss_four-girls.jpg)          | ![result](media/results/parrots-kiss_kiss.jpg)    | ![result](media/results/parrots-kiss_toyokawa-bridge.jpg)            | ![result](media/results/parrots-kiss_tarantula-nebula.jpg)   | ![result](media/results/parrots-kiss_starry-night.jpg)        | ![result](media/results/parrots-kiss_scream.jpg)      |

_Various styles applied to content images_

## Conclusion

Over all, I'm very happy with how this project turned out. Not only did I learn many new things about the PyTorch library, but I was also able to improve my skills in working with convolutional neural networks and finetuning learning processes. 

Furthermore, I'm really glad about the end results. With a different GPU, one with more memory, I'd probably be able to make the pictures look even better, as it would allow for neural style transfer with larger images.

---

... MattMoony (September, 2019)