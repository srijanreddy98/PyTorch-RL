import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# disable bundler
from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
# enable bundler

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("ggplot")
from ray import tune
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()
space = {
    "learning-rate": hp.uniform("learning-rate", 1e-10, 0.1),
    "gamma": hp.uniform("gamma", 0.90, 0.99),
}
hyperopt_search = HyperOptSearch(
    space, metric="mean_accuracy", mode="max")
def train_agent(config):
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    """environment"""
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    running_state = ZFilter((state_dim,), clip=5)
    # running_reward = ZFilter((1,), demean=False, clip=10)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    """define actor and critic"""
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
        value_net = Value(state_dim)
    policy_net.to(device)
    value_net.to(device)

    # optimization epoch number and batch size for PPO
    optim_epochs = 10
    optim_batch_size = 64

    """create agent"""
    agent = Agent(env, policy_net, device, running_state=running_state, render=args.render, num_threads=args.num_threads)



    def update_params(batch, i_iter, optimizer_policy, optimizer_value, config):
        print('1')
        states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
        actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
        print('2')
        with torch.no_grad():
            values = value_net(states)
            fixed_log_probs = policy_net.get_log_prob(states, actions)
        print('3')
        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, config['gamma'], args.tau, device)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
        print('5')
        for _ in range(optim_epochs):
            print(_)
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).to(device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                print(i)
                ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                        advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


    def main_loop(config):
        optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=config['learning-rate'])
        optimizer_value = torch.optim.Adam(value_net.parameters(), lr=config['learning-rate'])
        for i_iter in range(args.max_iter_num):
            """generate multiple trajectories that reach the minimum batch_size"""
            print(i_iter)
            batch, log = agent.collect_samples(args.min_batch_size)
            t0 = time.time()
            update_params(batch, i_iter,optimizer_policy, optimizer_value, config)
            t1 = time.time()
            print(i_iter)
            if i_iter % args.log_interval == 0:
                print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                    i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))
            tune.track.log(mean_accuracy=log['avg_reward'])
            tune.report(mean_accuracy=log['avg_reward'])

            if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
                to_device(torch.device('cpu'), policy_net, value_net)
                pickle.dump((policy_net, value_net, running_state),
                            open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
                to_device(device, policy_net, value_net)

            """clean up gpu memory"""
            torch.cuda.empty_cache()
    main_loop(config)


analysis = tune.run(
    train_agent,
    num_samples=1,  
    search_alg=hyperopt_search,
    verbose=1,
    name="cart_pole"
)
