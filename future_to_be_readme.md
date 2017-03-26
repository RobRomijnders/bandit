# Training Convolutional Neural nets with bandit feedback

This repo introduces the concepts of bandit feedback. In bandit feedback, the agent receives a reward for the action it took. However, it will never know what the reward are for other actions. Compare this to supervised learning: in SL, the agent gets a reward for the action it took and it knows that this action was the single best action to take. 

# Examples

  * Bandit feedback: (example from ad-placement) The agent receives features about a user and shows some advertisment. It can be rewarded with a click. Or the user does not click, which corresponds to reward=0
  * Supervision: (example from computer vision) The agent receives an image and labels it *cat*. If the label is wrong: the supervisor sends an error signal to increase the probability of the true label. If the label is correct: again, the supervisor sends an error signal to increase the probability of the true label. (assuming softmax distribution)

# Application

You can now imagine the wealth of applications:

  * Optimizing ad placement for media companies
  * Optimizing product recommendations for e-commerce companies
  * Optimizing contact recommendations for social networks 

# The project
We condense the problem to something we can visualize, explain and do on our laptop using only a cpu. The applications listed above involve major companies with big computational and engineering budget. This project aims at introducing the concepts. We use the Hello world! of machine
learning: MNIST. If we understand bandit algorithms on MNIST, we can readily extend this to other projects

# The problem
Think of recommendations:

  * We get features
  * We recommend some object
  * We observe if the user clicks/buys/links or not

 We simulate

   * the features as a mix of images of MNIST digits. These are our features
   * the recommendation is one of 9 products: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
   * the reward signal is if some part of the image corresponds to that image. (for simulation purposes, we inject some noise here)

## Technical details

 How is this different from supervised learning?

 Let's say our image contains a mix of numers 3, 7 and 8. 

   * The supervisor would say: the correct label is (3, 7, 8) any other label is incorrect. For every forward pass of the neural network, increase the probability of (3, 7, 8).
   * The bandit will say: you gave me a recommendation 7. I will click/buy number 7. That's a reward of +1. The agent will never known what the reward might have been for recommending 6. Or for recommending 3. It only gets a reward of +1 because 7 was a good recommendation.

# Model architecture

 Some text here

# Experiments

  Some text here

# Results

  Some text here

# Discussion

  Some text here
  m
# Conclusion
 

# Resources
 Some pointers for recources and further readings

   * Intuition on bandit feedback

       * [Ian Obands interactive demo](http://iosband.github.io/2015/07/28/Beat-the-bandit.html)

   * Learn on Counterfactual regret

       * [Lecture on logged bandit feedback at Microsoft Research](https://www.youtube.com/watch?v=4I0zsPTZyP4)
       * [Lecture notes **Introduction to Counterfactual regret**](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf)
       
   * Academic material

       * [Counterfactual risk minimzation](https://arxiv.org/pdf/1502.02362.pdf)
       * [Deepstack: Expert-Level Artificial Intelligence in No-Limit Poker](https://arxiv.org/abs/1701.01724)
