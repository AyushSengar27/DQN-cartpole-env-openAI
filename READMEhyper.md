# To run experimentation : python experiment_hyper.py --experiments
# To run individual models : python experiment_hyper.py --dqn (this will run dqn without TN and ER)
# python experiment_hyper.py --dqn --tn (this will run dqn with TN )
# python experiment_hyper.py --dqn --er (this will run dqn with ER )
# python experiment_hyper.py --dqn --er --tn (this will run dqn with ER and TN)


# Note : when you run the code you'll see green lines(a lot) on the terminal which indicate the epochs
# The code can take a lot of hours to run as it uses CPU version of tensorflow, for reference runninng the whole hyperprameter tuning code took 24+ hours on a 16gb windows machine. 