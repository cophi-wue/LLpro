# <p align=center>Scene Segmentation through Sequential Sentence Classification</p>
This repo has code for our paper ["Breaking the Narrative: Scene Segmentation through Sequential Sentence Classification"](http://lsx-events.informatik.uni-wuerzburg.de/files/stss2021/proceedings/kurfali_wiren.pdf) that ranked 1st at the [Shared Task on Scene Segmentation (STSS)](http://lsx-events.informatik.uni-wuerzburg.de/stss-2021/).

### How to run

The easist way to run our model on your data is through Docker. Please put the novels you want segmented into the "data/test" directory following  [the shared task's JSON format](http://lsx-events.informatik.uni-wuerzburg.de/stss-2021/task.html).

Then you can simply build & run the docker image by simply running:
```
./build.sh
./run.sh
```
The built image will be ~9 GB and the predictions will be saved in "predictions" folder: One output file will be created per input file with the same name.

### Citing

If you use the data or the model, please cite,
```
@inproceedings{kurfali2021breaking,
  title={Breaking the Narrative: Scene Segmentation through Sequential Sentence Classification},
  author={Kurfal{\i}, Murathan and Wir{\'e}n, Mats}
  journal={Shared Task on Scene Segmentation},
  year={2021}
}
```
