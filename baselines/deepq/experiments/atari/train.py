import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import tempfile
import time
import json
import pickle

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, ReplayBufferSp
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines.deepq.experiments.atari.model import model, dueling_model

def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default=None, help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Comments by Fei: about environment, are we in test mode with a test_path?
    parser.add_argument("--test_path", type=str, default=None, help="if in the test mode, give the directory of SAT problems for testing")
    parser.add_argument("--dump_pair_into", type=str, default=None, help="if in the test mode, give the directory of saving state-action pairs")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(5e6), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=40000, help="number of iterations between every target network update")
    parser.add_argument("--param-noise-update-freq", type=int, default=50, help="number of iterations between every re-scaling of the parameter noise")
    parser.add_argument("--param-noise-reset-freq", type=int, default=10000, help="maximum number of steps to take per episode before re-perturbing the exploration policy")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    boolean_flag(parser, "param-noise", default=False, help="whether or not to use parameter space noise for exploration")
    boolean_flag(parser, "layer-norm", default=False, help="whether or not to use layer norm (should be True if param_noise is used)")
    boolean_flag(parser, "gym-monitor", default=False, help="whether or not to use a OpenAI Gym monitor (results in slower training due to video recording)")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6, help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4, help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
    parser.add_argument("--keep_prob", type=float, default=1.0, help="the probably of keeping a hidden neuron in dropout")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=None, help="directory in which training state and model should be saved.")
    parser.add_argument("--model-dir", type=str, default=None, help="directory in which the model is saved for testing")
    parser.add_argument("--save-azure-container", type=str, default=None,
                        help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
    parser.add_argument("--save-freq", type=int, default=5e5, help="save model once every time this many iterations are completed")
    parser.add_argument("--model-rename-freq", type=int, default=5e5, help="save model at different name once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    return parser.parse_args()


def make_env(game_name):
    # env = gym.make(game_name + "NoFrameskip-v4")
    env = gym.make(game_name)
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    # env = wrap_dqn(monitored_env)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    # Comments by Fei: not needed for sat
    return monitored_env, monitored_env


def maybe_save_model(savedir, container, state):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = "model-{}".format(state["num_iters"] // args.model_rename_freq) # Comments by Fei: do some thing to make the name not so different (don't want to save too many models)
    U.save_state(os.path.join(savedir, model_dir, "saved"))
    if container is not None:
        container.put(os.path.join(savedir, model_dir), model_dir)
    relatively_safe_pickle_dump(state, os.path.join(savedir, 'training_state.pkl.zip'), compression=True)
    if container is not None:
        container.put(os.path.join(savedir, 'training_state.pkl.zip'), 'training_state.pkl.zip')
    relatively_safe_pickle_dump(state["monitor_state"], os.path.join(savedir, 'monitor_state.pkl'))
    if container is not None:
        container.put(os.path.join(savedir, 'monitor_state.pkl'), 'monitor_state.pkl')
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir, container):
    """Load model if present at the specified path."""
    if savedir is None:
        return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    if container is not None:
        logger.log("Attempting to download model from Azure")
        found_model = container.get(savedir, 'training_state.pkl.zip')
    else:
        found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model-{}".format(state["num_iters"] // args.model_rename_freq) # Comments by Fei: for whatever change in maybe_save_model, reflect it here!
        if container is not None:
            container.get(savedir, model_dir)
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state
    else:
        model_dir = args.model_dir
        if model_dir is not None:
            U.load_state(os.path.join(savedir, model_dir, "saved"))
        return None 

"""
    this function test the performance of the current deepq neural networks. 
    if dump_pair_into is provided, it will also dump state-action pair into the given directory
"""
import scipy.sparse as sp
def test_it(test_path):
    # Comments by Fei: specialized import (not supposed to be public)
    from gym.envs.SatSolver import gym_sat_Env, gym_sat_sort_Env, gym_sat_permute_Env, gym_sat_graph_Env, gym_sat_graph2_Env
    env_type = args.env
    if env_type == "gym_sat_Env-v0": 
        env = gym_sat_Env(test_path = test_path)
    elif env_type == "gym_sat_Env-v1": 
        env = gym_sat_sort_Env(test_path = test_path)
    elif env_type == "gym_sat_Env-v2": 
        env = gym_sat_permute_Env(test_path = test_path)
    elif env_type == "gym_sat_Env-v3":
        env = gym_sat_graph_Env(test_path = test_path)
    elif env_type == "gym_sat_Env-v4":
        env = gym_sat_graph2_Env(test_path = test_path)
    else: 
        print("ERROR: env is not one of the pre-defined mode")
        return

    test_file_num = env.test_file_num
    print("there are {} files to test".format(test_file_num))
    savedir = args.save_dir
    if savedir is None:
        savedir = os.getenv('OPENAI_LOGDIR', None)
    container = None
    with U.make_session(1) as sess:
        # initialize training graph 
        def model_wrapper(img_in, num_actions, scope, **kwargs):
            actual_model = dueling_model if args.dueling else model
            return actual_model(img_in, num_actions, scope, layer_norm=args.layer_norm, **kwargs)
        act, train, update_target, debug = deepq.build_train(
            # make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            make_obs_ph=lambda name: U.Int8Input(env.observation_space.shape, name=name), # Change for sat env
            q_func=model_wrapper,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=args.double_q,
            param_noise=args.param_noise
        )
        
        # load the model
        U.initialize()
        state = maybe_load_model(savedir, container)

        # preparation if we want to dump state-action pair
        dumpdir = args.dump_pair_into
        if dumpdir is None: 
            doDump = False
        else:
            doDump = True
            stateList = []
            actionList = []
        
        # main testing loop (measure the average steps needed to solve it)
        update_eps = 0.00 # all steps should be deterministic by the network we have!
        score = 0.0
        reward = 0
        kwargs = {}
        for i in range(test_file_num):
            obs = env.reset() # this reset is in test mode (because we passed test_path at the construction of env)
                              # so the reset will iterate all test files in test_path, instead of randomly picking a file
            while True:
                action, q_values = act(np.array(obs)[None], update_eps=update_eps, **kwargs)
                action = action[0]
                
                # if we want to dump state-action pair, here is the chance (np.array(obs)[None], action)
                if doDump:
                    # Comments by Fei: append state in sparse representation!
                    obs_shape = obs.shape
                    assert len(obs_shape) == 3, "observations is not 3-D"
                    assert obs_shape[2] == 1, "observations 3rd dimension is not of size 1"
                    newobs = sp.csc_matrix(obs[:,:,0])
                    stateList.append(newobs) 
                    # Comments by Fei: append action as q_values, not just the final choice
                    q_values_shape = q_values.shape
                    assert len(q_values_shape) == 2, "q_values is not 2-D"
                    assert q_values_shape[0] == 1, "q_values 1st dimension is not of size 1"
                    actionList.append(q_values[0,:])

                new_obs, rew, done, info = env.step(action)
                obs = new_obs
                reward += 1
                if done:
                    score = (score * i + reward) / (i+1)
                    reward = 0
                    break
        if doDump:
            with open(dumpdir, "wb") as f:
                pickle.dump({"states": stateList, "actions": actionList}, f)
                print("dump data in {}, total number is {}".format(dumpdir, len(actionList)))
        print("the average performance is {}".format(score))

if __name__ == '__main__':
    args = parse_args()
    
    # Comments by Fei: if we are in the test mode (test_path is not None), call test_it() function:
    if not args.test_path == None:
        test_it(args.test_path)
        exit(0)
    
    # Comments by Fei: if test_path is None, go ahead and train the model
    # Parse savedir and azure container.
    savedir = args.save_dir
    if savedir is None:
        savedir = os.getenv('OPENAI_LOGDIR', None)
    # Comments by Fei: sat solver will probably never use container option. Most importantly, servers I used didn't install azure! big trouble
    container = None
#    if args.save_azure_container is not None:
#        account_name, account_key, container_name = args.save_azure_container.split(":")
#        container = Container(account_name=account_name,
#                              account_key=account_key,
#                              container_name=container_name,
#                              maybe_create=True)
#        if savedir is None:
#            # Careful! This will not get cleaned up. Docker spoils the developers.
#            savedir = tempfile.TemporaryDirectory().name
#    else:
#        container = None
    # Create and seed the env.
    env, monitored_env = make_env(args.env)

    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    if args.gym_monitor and savedir:
        env = gym.wrappers.Monitor(env, os.path.join(savedir, 'gym_monitor'), force=True)

    if savedir:
        with open(os.path.join(savedir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    with U.make_session(4) as sess:
        # Create training graph and replay buffer
        def model_wrapper(img_in, num_actions, scope, **kwargs):
            actual_model = dueling_model if args.dueling else model
            return actual_model(img_in, num_actions, scope, layer_norm=args.layer_norm, keep_prob = args.keep_prob, **kwargs)
        act, train, update_target, debug = deepq.build_train(
            # make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            make_obs_ph=lambda name: U.Int8Input(env.observation_space.shape, name=name), # Change for sat env
            q_func=model_wrapper,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            gamma=0.99,
            grad_norm_clipping=10,
            double_q=args.double_q,
            param_noise=args.param_noise
        )

        approximate_num_iters = args.num_steps
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (approximate_num_iters / 50, 0.1),
            (approximate_num_iters / 5, 0.01)
        ], outside_value=0.01)

        if args.prioritized:
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        else:
            # Comments by Fei: changed to ReplayBufferSp to save space of saved transactions
            replay_buffer = ReplayBufferSp(args.replay_buffer_size)

        U.initialize()
        update_target()
        num_iters = 0

        # Load the model
        state = maybe_load_model(savedir, container)
        if state is not None:
            num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
            monitored_env.set_state(state["monitor_state"])

        start_time, start_steps = None, None
        steps_per_iter = RunningAvg(0.999)
        iteration_time_est = RunningAvg(0.999)
        obs = env.reset()
        num_iters_since_reset = 0
        reset = True

        # Main trianing loop
        while True:
            num_iters += 1
            num_iters_since_reset += 1

            # Take action and store transition in the replay buffer.
            kwargs = {}
            if not args.param_noise:
                update_eps = exploration.value(num_iters)
                update_param_noise_threshold = 0.
            else:
                if args.param_noise_reset_freq > 0 and num_iters_since_reset > args.param_noise_reset_freq:
                    # Reset param noise policy since we have exceeded the maximum number of steps without a reset.
                    reset = True

                update_eps = 0.01  # ensures that we cannot get stuck completely
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(num_iters) + exploration.value(num_iters) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = (num_iters % args.param_noise_update_freq == 0)

            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0][0]
            reset = False
            new_obs, rew, done, info = env.step(action)
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            if done:
                num_iters_since_reset = 0
                obs = env.reset()
                reset = True

            if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                    num_iters % args.learning_freq == 0):
                # Sample a bunch of transitions from replay buffer
                if args.prioritized:
                    experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                    weights = np.ones_like(rewards)
                # Minimize the error in Bellman's equation and compute TD-error
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                # Update the priorities in the replay buffer
                if args.prioritized:
                    new_priorities = np.abs(td_errors) + args.prioritized_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if start_time is not None:
                steps_per_iter.update(info['steps'] - start_steps)
                iteration_time_est.update(time.time() - start_time)
            start_time, start_steps = time.time(), info["steps"]

            # Save the model and training state.
            if num_iters > 0 and (num_iters % args.save_freq == 0 or info["steps"] > args.num_steps):
                maybe_save_model(savedir, container, {
                    'replay_buffer': replay_buffer,
                    'num_iters': num_iters,
                    'monitor_state': monitored_env.get_state(),
                })
                # Comments by Fei: tabular log only for each save
                steps_left = args.num_steps - info["steps"]
                completion = np.round(info["steps"] / args.num_steps, 1)

                logger.record_tabular("% completion", completion)
                logger.record_tabular("td_errors", td_errors) # Comments by Fei: why not report the error as well?
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))
                logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-100:]))
                logger.record_tabular("exploration", exploration.value(num_iters))
                if args.prioritized:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                fps_estimate = (float(steps_per_iter) / (float(iteration_time_est) + 1e-6)
                                if steps_per_iter._value is not None else "calculating...")
                logger.dump_tabular()
                logger.log()
                logger.log("ETA: " + pretty_eta(int(steps_left / fps_estimate)))
                logger.log()

            if info["steps"] > args.num_steps:
                break