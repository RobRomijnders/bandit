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

   * the features as the pixels of MNIST digits.
   * the recommendation is one of 9 products: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
   * the reward signal is if the recommendation corresponds to the actual digit in the image

## Technical details

 How does this differ from supervised learning?

 Let's the pixels in some image correspond to the digit 8

   * The supervisor would say: the correct label is (8) any other label is incorrect. For every forward pass of the neural network, increase the probability of (8).
   * The bandit will say: you gave me a recommendation 7. I will not click/buy number 7. That's a reward of 0. The agent will never known what the reward might have been for recommending 8. Or for recommending 3. It only gets a reward of 0 because 7 was a bad recommendation.

# Why do we consider policies?
When working with bandit problems, we'll refer to your recommendations as samples from a policy. What's a policy? A policy makes a distribution over possible actions given your current features. For example, a policy might be a conv-net with inputs an MNIST digit. Then the output would be a softmax distribution over the possible recommendations to make.

You might think: why does the policy output a distribution and not just a single recommendation? Good question. We want to sample a recommendation, such that we can learn from the user. If, for a given images, we'd always recommend 7, then how would we ever learn to get the best reward?

## Technical detail on the policy
We work with logged training data. For example your e-commerce website has been recommending books for years now. You want to use that information. But there's two catches here:

  * __Catch 1__: The logged data is incomplete. Every customers visits the website, you recommend him a book and he clicks or not. You'll never be able to know if he would have clicked on other books. Solutions to such problems are said to be __counterfactual__
  * __Catch 2__: The logged data is biased. Customers will more likely click on books that you show them on the home page. Say the website sells both novels and thrillers. If the homepage displays 100 thrillers and 20 novels, then you'll have many more logs on the thrillers. Such bias can be dealt with by importance sampling.

These catches come together in training our policy. The code in _model/conv.py_ constructs our policy in Tensorflow. Near line 60, we have
```python
importance = tf.divide(pol_y,pol_0)
importance_sampling = tf.multiply(importance,reward) 
```
We are _learning_ the ```pol_y```. Yet the website used ```pol_0``` during logging. The intuition for this formula goes like this:
_Say some log (features, action, reward) gives a reward of +8. Our old policy choose that action with a ```p=0.1```. Our new policy chooses the action with ```p=0.3```. Effectively, we reward the system with +24. However, if the old ```p=0.1``` and the new ```p=0.05```, then we effectively reward the system with +4. So every trainstep scales rewards by the **importance** of that log_


# The repo

  * ```main.py``` loads the modules and trains the policy
  * ```conv.py``` implement the convolutional neural network as our policy
  * ```intro_to_cfr/``` implements [chapter 3 of an intro to Counterfactual learning](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf). It ghelped me to understand counterfactual learning. That folder contains its own readme.
 

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
