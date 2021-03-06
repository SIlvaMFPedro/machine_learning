# Autonomous Vehicle Training for Obstacle Avoidance Using Reinforcement Learning

- This repository is related to the final project of machine learning which was created to learn the basics of reinforcment learning.
- It employes a Q-learning (unsupervised) algorithm to learn how to move an object around a screen (drive itself) without running into obstacles.
- The purpose of this project is to eventually use the learning algorithms from the game to operate a real-life remote-control car, using distance sensors(e.g: ATLASCAR).

# Dependencies
- Python3
- Pygame
- Pymunk
- Keras
- Theanos

# Install Instructions

These instructions are for a fresh Ubuntu 18.04LTS box. Most of the same should apply to OS X. If you have issues installing, feel free to open an issue with your error and I'll do my best to help.

# Basics
Recent Ubuntu releases come with python3 installed. I use pip3 for installing dependencies so install that with `sudo apt-get install python3-pip`. Install git if you don't already have it with `sudo apt-get install git`.

Then clone this repo with `git clone https://github.com/SIlvaMFPedro/machine_learning.git`. It has some pretty big weights files saved in past commits, so to just get the latest the fastest, do `git clone https://github.com/SIlvaMFPedro/machine_learning.git --depth 1`.


# Python Dependencies
Use this command to install all relevant python dependencies:
`pip3 install numpy keras h5py`

# Install Pygame

Install Pygame's dependencies with:

    sudo apt install mercurial libfreetype6-dev libsdl-dev libsdl-image1.2-dev libsdl-ttf2.0-dev 
    libsmpeg-dev libportmidi-dev libavformat-dev libsdl-mixer1.2-dev libswscale-dev libjpeg-dev

Then install Pygame itself:

    sudo pip3 install hg+http://bitbucket.org/pygame/pygame

# Install Pymunk

This is the physics engine used by the simulation. It just went through a pretty significant rewrite (v5) so you need to grab the older v4 version. v4 is written for Python 2 so there are a couple extra steps.

Go back to your home or downloads and get Pymunk 4:

    wget https://github.com/viblo/pymunk/archive/pymunk-4.0.0.tar.gz

Unpack it:

    tar zxvf pymunk-4.0.0.tar.gz

Update from Python 2 to 3:

    cd pymunk-pymukn-4.0.0/pymunk
    2to3 -w *.py

Install it:

    cd .. 
    python3 setup.py install

Now go back to the reinforcement_auto_pilot directory and make sure everything worked with a quick `python3 learn_model.py`. If you see a screen come up with a little dot flying around the screen, you're ready to go!


# Training Instructions
First, you need to train a model. This will save weights to the saved-models folder. You may need to create this folder before running. You can train the model by running:

    python3 learn_model.py

It can take anywhere from an hour to 36 hours to train a model, depending on the complexity of the network and the size of your sample. However, it will spit out weights every 25,000 frames, so you can move on to the next step in much less time.

# Playing Instructions
First, edit the `play_model.py` file to change the path name for the model you want to load. 
Then, watch the car drive itself around the obstacles!

    python3 play_model.py

That's all there is to it.

# Plotting Instructions

Once you have a bunch of CSV files created via the learning, you can convert those into graphs by running:

    python3 plot_model.py

This will also spit out a bunch of loss and distance averages at the different parameters.

# Videos

[![watch video play model](https://imgur.com/a/obXCW5k)](https://youtu.be/I3qTrub-1HQ)
[![watch video learn model](https://imgur.com/a/obXCW5k)](https://youtu.be/3R-EZr5wHkg)
# Credits

I'm grateful to the following people and the work they did that helped me learn how to do this:

- Playing Atari with Deep Reinforcement Learning - http://arxiv.org/pdf/1312.5602.pdf
- Deep learning to play Atari games: https://github.com/spragunr/deep_q_rl
- Another deep learning project for video games: https://github.com/asrivat1/DeepLearningVideoGames
- A great tutorial on reinforcement learning that a lot of my project is based on: http://outlace.com/Reinforcement-Learning-Part-3/