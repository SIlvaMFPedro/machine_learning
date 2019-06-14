#!/usr/bin/env python3

#
# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

# ------------------------
#   IMPORTS
# ------------------------
import numpy as np
import random
import csv
import os.path
import timeit
import car_module
from neural_network import neural_network, LossHistory

NUM_INPUT = 3
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.


# -----------------------------
#   Params to Filename Function
# -----------------------------
# -----------------------------
def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])


# -------------------------------
#   Train Neural Network Function
# -------------------------------
def train_network(model, params):
    filename = params_to_filename(params)
    observe = 1000  # number of frames needed to observe before starting the training process.
    epsilon = 1
    train_frames = 100000 # number of frames to be play during the training process.
    batchSize = params['batchSize']
    buffer = params['buffer']

    X_train = 0
    Y_train = 0
    X_test = 0
    Y_test = 0
    max_car_distance = 0
    car_distance = 0
    t = 0
    data_collect = []
    replay = []         # stores tuples of (S, A, R, S')
    loss_log = []

    game_state = car_module.GameState()     # create new game instance
    _, state = game_state.frame_step(2)     # get initial state from the vehicle object.
    start_time = timeit.default_timer()     # start timer

    # Run Frames
    while t < train_frames:
        t += 1
        car_distance += 1
        # choose an action
        if random.random() < epsilon or t < observe:
            action = np.random.randint(0, 3)
        else:
            # get Q values for each one of the actions
            qval = model.predict(state, batch_size = 1)
            action = (np.argmax(qval))  # save best value of qval

        # take the action, observe new state and get the treat
        reward, new_state = game_state.frame_step(action)
        # save experience in replay storage
        replay.append((state, action, reward, new_state))

        # start the training process after observing
        if t > observe:
            # if there is no more space in the buffer, pop the oldest sample
            if len(replay) > buffer:
                replay.pop(0)
            # pick a random sample experience from replay memory
            minibatch = random.sample(replay, batchSize)
            # get test values
            x_test, y_test = process_minibatch_test(minibatch, model)
            X_test, Y_test = x_test, y_test
            # get training values
            x_train, y_train = process_minibatch_train(minibatch, model)
            X_train, Y_train = x_train, y_train
            # train the model on this batch
            history = LossHistory()
            model.fit(x_train, y_train, batch_size=batchSize,
                      epochs=1, verbose=0, callbacks=[history])
            loss_log.append(history.losses)

        # update current state
        state = new_state

        # decrement epsilon value over time
        if epsilon > 0.1 and t > observe:
            epsilon -= (1.0/train_frames)
        # if the vehicle dies, update car distance parameters
        if reward == -500:
            # log the current car's distance
            data_collect.append([t, car_distance])
            # update max distance
            if car_distance > max_car_distance:
                max_car_distance = car_distance
            # set timer
            tot_time = timeit.default_timer() - start_time
            fps = car_distance / tot_time
            # print results
            print("Max: %d at %d\tepsilon %f\t(%d)\t%f fps" % (max_car_distance, t, epsilon, car_distance, fps))
            # reset values
            car_distance = 0
            start_time = timeit.default_timer()

        # save the model every 25000 frames
        if t % 25000 == 0:
            model.save_weights('saved-models/' + filename + '-' +
                               str(t) + '.h5',
                               overwrite=True)
            print("Saving model %s - %d" % (filename, t))
            evaluate_network(model, X_train, Y_train, batchSize)
            test_network(model, X_test, Y_test, batchSize)

        # save results in a log file
        log_results(filename, data_collect, loss_log)


# ----------------------------------
#   Evaluate Neural Network Function
# ----------------------------------
def evaluate_network(model, x_train, y_train, batchSize):
    score = model.evaluate(x_train, y_train, batch_size=batchSize)
    print("Evaluation Score: %d" % score)


# ----------------------------------
#   Test Neural Network Function
# ----------------------------------
def test_network(model, x_test, y_test, batchSize):
    score = model.evaluate(x_test, y_test, batch_size=batchSize)
    print("Test Score: %d" % score)


# -------------------------------
#   Log Results Function
# -------------------------------
def log_results(filename, data_collect, loss_log):
    # Save the results to a file so they can later be graphed.
    with open('results/frames/learn_data-' + filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results/frames/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)


# -------------------------------
#   Process Minibatch Function
# -------------------------------
def process_minibatch_train(minibatch, model):
    # Instead of feeding the data to the model one by one, feeding the whole batch proved to be much more efficient
    mb_len = len(minibatch)

    old_states = np.zeros(shape=(mb_len, 3))
    actions = np.zeros(shape=(mb_len,))
    rewards = np.zeros(shape=(mb_len,))
    new_states = np.zeros(shape=(mb_len, 3))

    for i, m in enumerate(minibatch):
        old_state_m, action_m, reward_m, new_state_m = m
        old_states[i, :] = old_state_m[...]
        actions[i] = action_m
        rewards[i] = reward_m
        new_states[i, :] = new_state_m[...]

    old_qvals = model.predict(old_states, batch_size=mb_len)
    new_qvals = model.predict(new_states, batch_size=mb_len)

    maxQs = np.max(new_qvals, axis=1)
    y = old_qvals
    non_term_inds = np.where(rewards != -500)[0]
    term_inds = np.where(rewards == -500)[0]

    y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (GAMMA * maxQs[non_term_inds])
    y[term_inds, actions[term_inds].astype(int)] = rewards[term_inds]

    x_train = old_states
    y_train = y
    return x_train, y_train


# -------------------------------
#   Process Minibatch Function
# -------------------------------
def process_minibatch_test(minibatch, model):
    # This function feeds the data to the model one by one
    x_test = []
    y_test = []
    # Loop through the batch and create arrays for x and y so that we can fit the model at every step
    for memory in minibatch:
        old_state_m, action_m, reward_m, new_state_m = memory   # get stored values
        old_qval = model.predict(old_state_m, batch_size=1)     # get prediction on old state
        newQ = model.predict(new_state_m, batch_size=1)         # get prediction on new state
        maxQ = np.max(newQ)                                     # get the predicted best move
        y = np.zeros((1, 3))
        y[:] = old_qval[:]
        # check for terminal state
        if reward_m != -500:                                    # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        else:                                                   # terminal state
            update = reward_m
        # update the value for the taken action
        y[0][action_m] = update
        x_test.append(old_state_m.reshape(NUM_INPUT, ))
        y_test.append(y.reshape(3, ))

    x_train = np.array(x_test)
    y_train = np.array(y_test)
    return x_train, y_train


# -------------------------------
#   Launch Learning Function
# -------------------------------
def launch_learn(params):
    filename = params_to_filename(params)
    print("Trying %s" % filename)
    # Make sure we haven't run this one.
    if not os.path.isfile('results/frames/loss_data-' + filename + '.csv'):
        # Create file so we don't double test when we run multiple
        # instances of the script at the same time.
        open('results/frames/loss_data-' + filename + '.csv', 'a').close()
        print("Starting test.")
        # Train.
        model = neural_network(NUM_INPUT, params['nn'])
        train_network(model, params)
    else:
        print("Already tested.")


# -------------------------------
#   Main Function
# -------------------------------
if __name__ == "__main__":
    if TUNING:
        param_lst = []
        nn_params = [[164, 150], [256, 256],
                     [512, 512], [1000, 1000]]
        batchSizes = [40, 100, 400]
        buffers = [10000, 50000]

        for nn_param in nn_params:
            for batchSize in batchSizes:
                for buff in buffers:
                    params = {
                        "batchSize": batchSize,
                        "buffer": buff,
                        "nn": nn_param
                    }
                    param_lst.append(params)

        for param_set in param_lst:
            launch_learn(param_set)

    else:
        nn_param = [128, 128]
        params = {
            "batchSize": 64,
            "buffer": 50000,
            "nn": nn_param
        }
        model = neural_network(NUM_INPUT, nn_param)
        train_network(model, params)