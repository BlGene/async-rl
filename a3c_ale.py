from __future__ import print_function
import argparse
import copy
import multiprocessing as mp
import os
import sys
import statistics
import time

import chainer
from chainer import links as L
from chainer import functions as F
import numpy as np

import policy
import v_function
import dqn_head
import a3c
import ale
import random_seed
import async
import rmsprop_async
from prepare_output_dir import prepare_output_dir
from nonbias_weight_decay import NonbiasWeightDecay
from init_like_torch import init_like_torch
from dqn_phi import dqn_phi

from pdb import set_trace



class A3CFF(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions, seed):
        self.head = dqn_head.NIPSDQNHead()
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions, seed)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        if sys.version_info < (3,0):
            super(A3CFF, self).__init__(self.head, self.pi, self.v)
        else:
            super().__init__(self.head, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        return self.pi(out), self.v(out)


class A3CLSTM(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_actions, seed):
        self.head = dqn_head.NIPSDQNHead()
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions, seed)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        self.lstm = L.LSTM(self.head.n_output_channels,
                           self.head.n_output_channels)
        if sys.version_info < (3,0):
            super(A3CLSTM, self).__init__(self.head, self.lstm, self.pi, self.v)
        else:
            super().__init__(self.head, self.lstm, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            out = self.lstm(out)
            self.lstm.h, self.lstm.c = prev_h, prev_c
        else:
            out = self.lstm(out)
        return self.pi(out), self.v(out)

    def reset_state(self):
        self.lstm.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()


def eval_performance(rom, p_func, n_runs):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    scores = []
    for i in range(n_runs):
        env = ale.ALE(rom, treat_life_lost_as_terminal=False)
        test_r = 0
        while not env.is_terminal:
            s = chainer.Variable(np.expand_dims(dqn_phi(env.state), 0))
            pout = p_func(s)
            a = pout.action_indices[0]
            test_r += env.receive_action(a)
        scores.append(test_r)
        print('test_{}:'.format(i), test_r)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    return mean, median, stdev


def train_loop(process_idx, counter, max_score, args, agent, env, start_time):
    try:

        total_r = 0
        episode_r = 0
        global_t = 0
        local_t = 0

        while True:

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value
            local_t += 1

            if global_t > args.steps:
                break

            agent.optimizer.lr = (
                args.steps - global_t - 1) / args.steps * args.lr

            total_r += env.reward
            episode_r += env.reward

            action = agent.act(env.state, env.reward, env.is_terminal)

            if env.is_terminal:
                if process_idx == 0:
                    print('{} global_t:{} local_t:{} lr:{} episode_r:{}'.format(
                        args.outdir, global_t, local_t, agent.optimizer.lr, episode_r))
                episode_r = 0
                env.initialize()
            else:
                env.receive_action(action)

            if global_t % args.eval_frequency == 0:
                # Evaluation

                # We must use a copy of the model because test runs can change
                # the hidden states of the model
                test_model = copy.deepcopy(agent.model)
                test_model.reset_state()

                def p_func(s):
                    pout, _ = test_model.pi_and_v(s)
                    test_model.unchain_backward()
                    return pout
                mean, median, stdev = eval_performance(
                    args.rom, p_func, args.eval_n_runs)
                with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
                    elapsed = time.time() - start_time
                    record = (global_t, elapsed, mean, median, stdev)
                    print('\t'.join(str(x) for x in record), file=f)
                with max_score.get_lock():
                    if mean > max_score.value:
                        # Save the best model so far
                        print('The best score is updated {} -> {}'.format(
                            max_score.value, mean))
                        filename = os.path.join(
                            args.outdir, '{}.h5'.format(global_t))
                        agent.save_model(filename)
                        print('Saved the current best model to {}'.format(
                            filename))
                        max_score.value = mean

    except KeyboardInterrupt:
        if process_idx == 0:
            # Save the current model before being killed
            agent.save_model(os.path.join(
                args.outdir, '{}_keyboardinterrupt.h5'.format(global_t)))
            print('Saved the current model to {}'.format(
                args.outdir), file=sys.stderr)
        raise

    if global_t == args.steps + 1:
        # Save the final model
        agent.save_model(
            os.path.join(args.outdir, '{}_finish.h5'.format(args.steps)))
        print('Saved the final model to {}'.format(args.outdir))


def train_loop_with_profile(process_idx, counter, max_score, args, agent, env,
                            start_time):
    import cProfile
    cmd = 'train_loop(process_idx, counter, max_score, args, agent, env, ' \
        'start_time)'
    cProfile.runctx(cmd, globals(), locals(),
                    'profile-{}.out'.format(os.getpid()))


def write_episode(args, episode_num, envs, acts, rwds):
    import tempfile

    assert( len(envs) == len(acts))
    assert( len(acts) == len(rwds))

    #check that a recorded_episodes dir exists
    rec_dir = os.path.join( args.outdir , "recorded_episodes" )

    #check that there is no folder for our episode
    ep_path = os.path.join( rec_dir, str(episode_num).zfill(4) )

    if os.path.exists( ep_path ):
        ep_path = tempfile.mkdtemp( rec_dir )
    else:
        os.makedirs(ep_path)

    act_fn = os.path.join(ep_path,"act.log")
    with open(act_fn, "w") as act_fo:
        act_fo.writelines([str(x)+"\n" for x in acts])

    rwds_fn = os.path.join(ep_path,"rwds.log")
    with open(rwds_fn, "w") as rwds_fo:
        rwds_fo.writelines([str(x)+"\n" for x in rwds])

    from PIL import Image
    for i, env in enumerate(envs):
        image_fn = os.path.join(ep_path,str(i).zfill(5) + ".png")
        image = Image.fromarray( env )
        image.save( image_fn )


def record_loop(process_idx, counter, max_score, args, agent, env, start_time):

    # step counter
    global_t = 0
    local_t = 0

    global_e = 0

    while True:
        print( "Recording Episode: ", global_e, global_t, "/", args.steps, file=sys.stderr)
        sys.stderr.flush()

        # Evaluation
        env_record = []
        act_record = []
        rwd_record = []

        # We must use a copy of the model because test runs can change
        # the hidden states of the model
        test_model = copy.deepcopy(agent.model)
        test_model.reset_state()

        def p_func(s):
            pout, _ = test_model.pi_and_v(s)
            test_model.unchain_backward()
            return pout

        env = ale.ALE(args.rom, treat_life_lost_as_terminal=False)
        test_r = 0

        while not env.is_terminal:

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value
            local_t += 1

            if global_t > args.steps:
                break

            s = chainer.Variable(np.expand_dims(dqn_phi(env.state), 0))
            pout = p_func(s)
            a = pout.action_indices[0]
            rwd = env.receive_action(a)

            test_r += rwd

            env_record.append( env.ale.getScreenRGB() )
            act_record.append( a )
            rwd_record.append( rwd )

        write_episode( args, global_e, env_record, act_record, rwd_record)

        if global_t > args.steps:
            break

        global_e += 1

def main():

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--use-sdl', action='store_true')
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--use-lstm', action='store_true')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--record', type=int, default=None)

    parser.set_defaults(use_sdl=False)
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 2 ** 16)


    # I suggest using train_randstate instead of np.random because it proably
    # behaves better for async use.
    train_randstate = np.random.RandomState(args.seed)

    # Choose random seed before async execution, in oder to assure
    # that we obtain different seeds for each process. This can be checked
    # by making sure each emulator has different seed, this works because each
    # emulator is set to have the same random seeds as its process ( the ALE python
    # class ) see ale.py for detials
    process_seeds = train_randstate.randint(0, 2 ** 16, args.processes)

    args.outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    def model_opt(seed=args.seed):
        if args.use_lstm:
            model = A3CLSTM(n_actions,seed=seed)
        else:
            model = A3CFF(n_actions,seed=seed)
        opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        if args.weight_decay > 0:
            opt.add_hook(NonbiasWeightDecay(args.weight_decay))
        return model, opt

    shared_model, shared_opt = model_opt()

    # Load saved model, move this code to A3CModel and RMSpropAsync
    if args.weights is not None:
        from chainer import serializers
        model_filename = args.weights

        serializers.load_hdf5(model_filename, shared_model)
        print('Loaded model {0}'.format( args.weights ))

        opt_filename = model_filename + '.opt'
        if os.path.exists(opt_filename):
            serializers.load_hdf5(opt_filename, shared_opt)
        else:
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))


    shared_model_params = async.share_params_as_shared_arrays(shared_model)
    shared_opt_states = async.share_states_as_shared_arrays(shared_opt)

    max_score = mp.Value('f', np.finfo(np.float32).min)
    counter = mp.Value('l', 0)
    start_time = time.time()

    # Write a header line first
    with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        print('\t'.join(column_names), file=f)

    # convert np.int64 to python int for JSON
    process_seeds = [int(x) for x in process_seeds]

    def run_func(process_idx):
        env = ale.ALE(args.rom,
                      seed=process_seeds[process_idx],
                      use_sdl=args.use_sdl)

        local_model, local_opt = model_opt(seed=process_seeds[process_idx])
        async.set_shared_params(local_model, shared_model_params)
        async.set_shared_states(local_opt, shared_opt_states)

        agent = a3c.A3C(local_model, local_opt, args.t_max, 0.99, beta=args.beta,
                        process_idx=process_idx, phi=dqn_phi)

        if args.record:
            record_loop(process_idx, counter, max_score,
                      args, agent, env, start_time)

        elif args.profile:
            train_loop_with_profile(process_idx, counter, max_score,
                                    args, agent, env, start_time)
        else:
            train_loop(process_idx, counter, max_score,
                       args, agent, env, start_time)


    async.run_async(args.processes, run_func)
    # for debugging:
    # run_func(0)

if __name__ == '__main__':
    main()
