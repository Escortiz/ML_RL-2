"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym
from gym.wrappers import RecordVideo
import os

from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    video_folder = "./videos_hopper_gym021"
    os.makedirs(video_folder, exist_ok=True)  # Crear carpeta si no existe

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    # Configuración para la grabación de video
    if args.record_video:
        if not os.path.exists(args.video_folder):
            os.makedirs(args.video_folder)
        # Define quale episodi si registrano, si registra usando il arg record_every
        episode_trigger = lambda episode_id: (episode_id % args.record_every == 0)
        # Envolvemos el entorno base con RecordVideo
        # Si tu versión de gym necesita render_mode='rgb_array', ajústalo al crear el env, para poder grabar
        # Gym lo suele manejar automatico, pero si hay que especificar:
        # env_base = gym.make('CustomHopper-source-v0', render_mode='rgb_array')
        env = RecordVideo(env, video_folder=args.video_folder, episode_trigger=episode_trigger,
                          name_prefix="test-agent-episode")
        print(f"Recording videos to {args.video_folder}, every {args.record_every} episode(s).")

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())
	
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(args.model), strict=True)

    agent = Agent(policy, device=args.device)

    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:

            action, _ = agent.get_action(state, evaluation=True)

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.render:
                env.render()

            test_reward += reward

        print(f"Episode: {episode} | Return: {test_reward}")
	

if __name__ == '__main__':
    main()