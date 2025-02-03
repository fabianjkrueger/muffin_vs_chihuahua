![Chihuahua or Muffin?](images/title_image.jpg)

# Muffin vs. Chihuahua

## Project Description

### Problem and Origin

There is a semi popular meme about the claim that computer vision models have
difficulties distinguishing models from chihuahuas.

<p align="center">
  <img src="images/muffin-meme2.jpg"
        width="50%"
        alt="Muffin or Chihuahua Meme">
</p>

As you can see, the muffins and the dogs do in fact look similar.
If the photo was taken in the right angle, and it isn't hard to imagine a
computer may have difficulties distinguishing between them.

For us humans it is easy to distinguish animals from food, because we can rely
on intuition, but a computer vision model has to learn the subtle differences
of the patterns.
Regarding their shape, spatial arrangement, texture and color, images of these
objects can be very similar.

One example of an early computer vision model actually misclassifying a cookie
for a pet was found when testing Google's Cloud Vision API in 2016:

<p align="center">
  <img src="images/misclassification_googlecloudvision2016.png"
       width="50%"
       alt="Google Cloud Vision API Misclassification">
</p>

But the question is if modern computer vision systems actually still
struggle with this or if there is no truth to the meme anymore.
Honestly, I am aware this problem is solved by now,
but I think it's still worth exploring this meme using just a laptop
and models you can fine-tune yourself.

There are even more examples like the one with the muffins and chihuahuas.
For example, this one in which poodles look like fried chicken.

<p align="center">
  <img src="images/Fried_Chicken_or_Labradoodle.jpg"
       width="50%"
       alt="Fried Chicken Meme">
</p>

Or these ones even including different animals like kittens and birds.

<p align="center">
  <table width="50%"
  align="center"
  border="0"
  cellspacing="0"
  cellpadding="0">
    <tr>
      <td><img src="images/owl_vs_apple.jpg"
      width="100%"
      alt="Barn Owl or Apple"></td>
      <td><img
      src="images/Kitten_or_Ice_Cream.jpg"
      width="100%"
      alt="Kitten or Ice Cream"></td>
    </tr>
    <tr>
      <td><img src="images/Shiba_Inu_or_Marshmallow.jpg"
      width="100%"
      alt="Shiba Inu or Marshmallow"></td>
      <td><img src="images/Parrot_or_Guacamole.jpg"
      width="100%"
      alt="Parrot or Guacamole"></td>
    </tr>
  </table>
</p>


In this project, I want to explore this meme a little and see myself
if there is any truth to it.
Again, it is important to know that these memes arose in 2017.
While models in 2016 may have had difficulties with this,
modern models should be able to handle it.
Let's find out!

Also regarding the description of how the model may be used:

In the real world, I hardly doubt that the model which will be trained in this
project is going to be useful for anything.
I can hardly imagine a real world situation in which it may be used, or in which
modern and more general computer vision models wouldn't probably be able to do 
the job well enough.
I doubt that there is a serious computer program in need of a model able to
perform binary classification between muffins and chihuahuas.
This is more like an experiment or hobby research project to explore
different computer vision model's capacities and to get to the bottom of this
meme.

#### Further things to integrate in the README problem description









- [An article about the meme in 2016](
    https://www.mirror.co.uk/news/weird-news/muffins-chihuahuas-bizarre-picture-quiz-7539743
)
- [Steren's Labs tries to confuse Google's Vision algorithms with dogs and muffins](
    https://labs.steren.fr/2016/03/27/trying-to-confuse-googles-vision-algorithms-with-dogs-and-muffins/
)
- testing different AIs on this problem:
    - testing CloudSight (2017): classified it mostly correct
    as tested here: https://blog.cloudsight.ai/chihuahua-or-muffin-1bdf02ec1680
    - comprehensive test of six APIs (2017):
    mostly correct, also some failures.
    interestingly, the wrongly classified cases actually alighn with the meme,
    so there was a case of a muffin being classified as a dog snout.
    still, most of it was actually correct.
    https://www.topbots.com/chihuahua-muffin-searching-best-computer-vision-api/
    - testing Google's Cloud Vision API (2016):
    "For almost each set, there is one tile that is completely wrong, but the
    rest is at least in the good category.
    Overall, I am really surprised how well it performs."
    Basically the AI is mostly correct, but there are some misclassifications.
    Interestingly, they often align with the memes (as seen in example).
    https://labs.steren.fr/2016/03/27/trying-to-confuse-googles-vision-algorithms-with-dogs-and-muffins/
- based on this literature and these tests out there, it has not really been an
actual challenge for AI to distinguish between these things since at least 2016.
Even back then, the AI makes very few mistakes.
Still, interestingly, when it does make mistakes, they often align with the
meme, so apparently there is some truth to it, and the objects don't just look
similar to us humans.

#### There are even papers out there mentioning this precise problem
I'll just store them here for now, but will find a better place soon

##### https://arxiv.org/abs/1801.09573

- Title: "Deep Learning Approach for Very Similar Objects Recognition
Application on Chihuahua and Muffin Problem"
- They entitle this sort of problem as "very similar object recognition"
- The authors themselves consider this study to have solved the problem:
"The proposed method is fully solved the very similar object recognition like
muffin or Chihuahua. It is the right solution for the such problem."
- My opinion: It's a very cool project, and I love that it is documented.
It is not a peer-reviewed paper, however.
It is just in arXive.
Does that matter? For this fun / meme project no!
If I tried to conduct a proper scientific study myself, then perhaps yes.

##### https://aclanthology.org/2024.acl-long.370/

- Title: "Muffin or Chihuahua? Challenging Multimodal Large Language Models with
Multipanel VQA"
- This one is even peer reviewed and published in the legitimate journal
"ACL Anthology"

### Solution, Model and Results

I fine tuned a binary image classifier in PyTorch on the muffin vs chihuahua
data set.
It's a ResNet18 convolutional network.
It is not too bad so far, but full evaluation on the test set is pending.










## How to get the code

You can clone this repository from GitHub with this command:

```bash
git clone https://github.com/fabianjkrueger/muffin_vs_chihuahua
```

If you don't have git installed, you can get it
[here](https://git-scm.com/downloads).


## Environment

The environment is managed with conda.
To build and activate the environment, you need to have a conda installation
(such as conda, miniconda, mamba or miniforge).
If you don't have a conda installation, please install one first.
I recommend [miniforge](https://github.com/conda-forge/miniforge),
because it is an open source lightweight version of conda.
The dependencies are listed in the file `environment.yaml`.

**To build and activate the environment, just run this:**

```bash
# build the environment from environment.yaml file
conda env create -f environment.yaml

# activate the environment
conda activate muffin_vs_chihuahua
```

There is also a `requirements.txt` file in the `api` directory.
It is used to install the dependencies for the API within the docker container.
The dependencies there are reduced to the minimum to run the API, but they will
not be sufficient to run the training code or the notebooks.
So, please use the `environment.yaml` in conda to build and activate the
environment.

## Data

The data used for this project is from Kaggle.
It is a dataset of images of muffins and chihuahuas.
You can find it [here](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification/).

### Download the Data

The data is not included in the repository when you clone it from GitHub to
save space.
If you want to run the code yourself, you have to download the data.
There are multiple ways to do this.
For example, you can download it manually from the Kaggle website.
But I made it easy for you: There is a script in the `scripts` directory that
will download the data for you.
To download the data, just run the script `scripts/get_data.sh` in your
terminal and you're done good to go.
You can run it from any working directory, because paths are managed internally.

```bash
# run the script
bash scripts/get_data.sh
```

If you use a shell other than bash, you can also run the script with
`sh` or `zsh` or whatever you use, as long as it is a POSIX compatible shell.
If you're on Windows, you can use the Windows Subsystem for Linux (WSL) to run
bash scripts.

### Process the Data

Notebook: `notebooks/exploratory_data_analysis.ipynb`
Here, the images are resized to 224x224 pixels.
This step is necessary for the downstream training.
Beyond that, the data is explored and visualized.
While this is not necessary for you to reproduce the results,
it is useful for you to understand the data and the problem.

## How to train the model

This is the pipeline for training the model.
If you just want to use the model, you don't need to run this.
For that, you can just use the API as described in the next section.
If you want to reproduce the results, you can run the code in this order:

### Hyper Parameter Optimization
Notebook: `notebooks/training.ipynb`
This notebook conducts a hyper parameter optimization.
It is not necessary for you to reproduce the results,
but it is useful for you to understand the problem and the solution.
The best hyper parameters were already determined in the notebook and
applied to the training script.
This one runs quite a while, so I do not recommend running it.

### Train the model
The final model is saved to the `models` directory.
It is named `final_model.pt`.
You don't have to run the trainig logic, because you can just access the
model in the repo.

However, if you do want to train it yourself, use the script
`scripts/train.py`.
Just run it from the command line with `python scripts/train.py`.
This trains the final model with the best hyperparameters determined in the
previous step.
Output is logged to the terminal
and the file `logs/final_model_train.log`.

This also evaluates the model on the test set and logs the results to the
file `logs/final_model_test.log`.

## Deploy the model

Here, the model is hosted as an API with FastAPI.
It can be run locally in a docker container.
While FastAPI is included in the conda environment,
Docker must be installed on your machine to run the API.
If you don't have docker installed, you can get it
[here](https://www.docker.com/get-started/).

### Run the API

To run the API, you can use the following commands:

```bash
# build the docker image
docker build -t muffin_vs_chihuahua_api .

# run the docker image
docker run -p 8000:8000 muffin_vs_chihuahua_api
```

Documentation for the API can be found [here](http://localhost:8000/docs) once
it is running.

### Query the API

The API can be queried with a simple HTTP request.
I implemented a test case for you, so you can just host it and query it.
You can find it in the notebook `notebooks/test_api.ipynb`.
Just open that notebook and run the cells.
You will find further information and instructions in the notebook.


# TODO

- Finish texts in the training notebook
- Finalize description of the project
  - Problem is described in the README so it's clear what the problem is and how
  the solution will be used
- Make the notebook for testing the API a bit more user friendly
  - Save one or a few images to the `images` directory and load them from there
    so that the user can run it without having to download the entire dataset


- Check once again if everything works and is reproducible
  - Making the environment from the yaml file
  - Running the scripts
  - Running the notebooks
  - Running the API
  - Querying the API

- Funny part of the project
  - Split the meme into the individual images
  - Query the model with all of them and the combined one to see how it does
  - Declare the problem solved or not

# Sources

- [An article about the meme in 2016](
    https://www.mirror.co.uk/news/weird-news/muffins-chihuahuas-bizarre-picture-quiz-7539743
)
- [Steren's Labs tries to confuse Google's Vision algorithms with dogs and muffins](
    https://labs.steren.fr/2016/03/27/trying-to-confuse-googles-vision-algorithms-with-dogs-and-muffins/
)
- [CloudSight API tested on the meme](
    https://blog.cloudsight.ai/chihuahua-or-muffin-1bdf02ec1680
)
- [TopBots tests a wide range of APIs in 2017](
    https://www.topbots.com/chihuahua-muffin-searching-best-computer-vision-api/
)
