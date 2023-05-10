from gym.envs.box2d.car_racing import CarRacing
from agent_ppo import PPOAgent, Record
from helper_func import process_state_img, custom_reward

from colors import colors

epoch_num = 1000
# frame_num = 1000
epi_r_list = []


if __name__ == '__main__':
    # env = CarRacing(render_mode = 'human')
    env = CarRacing()
    agent_ppo = PPOAgent()

    for epoch in range(epoch_num):
        epi_reward = 0
        state, _ = env.reset()
        state = process_state_img(state)

        while True:
            action, a_logp = agent_ppo.get_action(state)
            next_state, reward, done, die, _ = env.step(action)
            # env.render()

            next_state = process_state_img(next_state)

            if agent_ppo.store(Record(state, action, a_logp, reward, next_state)):
                print('updating')
                print(colors.OKCYAN, "\naction", action, a_logp, colors.ENDC)
                print(epoch, " epi_reward: ", epi_reward)
                agent_ppo.learn()
            
            epi_reward = epi_reward + reward + custom_reward(state)
            state = next_state

            if epi_reward < -50: break
            
            if done:
                print('epi: ', epoch, ' epi_reward:', epi_reward, '\n')
            if done or die: break

        epi_r_list.append(epi_reward)
        
        if epoch and epoch % 100 == 0:
            agent_ppo.save_model(epoch)
            agent_ppo.save_loss(epi_r_list)


    agent_ppo.save_model(epoch_num)
    agent_ppo.save_loss(epi_r_list)
    env.close()
