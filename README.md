![Chihuahua or Muffin?](images/title_image.jpg)

# Muffin vs. Chihuahua

## Project Description

### Problem and Origin
There is a semi popular meme about the claim that computer vision models have
difficulties distinguishing models from chihuahuas. There are even more
examples like this, for example one in which poodles look like fried chicken.
As you can see, the muffins and the dogs do in fact look similar if the photo
was taken in the right angle, and it isn't hard to imagine a computer may have
difficulties distinguishing between them.

In this project, I want to explore this meme a little and see myself
if there is any truth to it.
It is also important to know that the meme arose in 2017.
Most likely, progress in the field made it much easier for modern models to
deal with this.
Let's find out!

Also regarding the description of how the model may be used: 
Like this model isn't even useful for anything.
I cannot imagine any situation in which it may be used.
I doubt that there is a serious computer program in need of a model able to do
perform binary classification between chihuahuas and muffins in the real world.
This is more like an experiment or rather research project to explore
different computer vision model's capacities and to get to the bottom of this
meme.

#### Further things to integrate in the README problem description

- "is a fun way to illustrate the limits of computer vision systems"
- include further pictures of animals vs food here
- state that this is probably a solved problem by now, but in 2016/7 it was real
- for us humans it is easy to distinguish between, because we can rely on 
intuition, but for a model it can be difficult. regarding their form and color,
these objects look very similar. AIs cannot rely on years of experience and
doesn't initially "know" the subtle differences giving away the class like
whiskers or the muffin cup
- this is a popular meme in the ML community, and it is often shared in
presentations. but the question is if modern computer vision systems actually
struggle with this or if there is no truth to it
- I could also test a non-fine-tuned ResNet18 for comparison and see what it
predicts. that way I can assess how much benefit training it brought. maybe it
is already good enough. however, there are much more classes, not just binary
- the meme came up in 2016: 
https://www.mirror.co.uk/news/weird-news/muffins-chihuahuas-bizarre-picture-quiz-7539743


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

## Environment
To build and activate the environment, just run this:
```bash
# build the environment from environment.yaml file
conda env create -f environment.yaml

# activate the environment
conda activate muffin_vs_chihuahua
```

## Data
To download the data, just run the script `scripts/get_data.sh` in the terminal.
You can run it from any working directory, because it manages the paths.


