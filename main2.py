import gym
from model2 import Actor_Critic
import matplotlib.pyplot as plt
import wandb

if __name__ == "__main__":
    def query_environment(name):
        env = gym.make(name)
        spec = gym.spec(name)
        print(f"Action Space: {env.action_space}")
        print(f"Observation Space: {env.observation_space}")
        print(f"Max Episode Steps: {spec.max_episode_steps}")
        print(f"Nondeterministic: {spec.nondeterministic}")
        print(f"Reward Range: {env.reward_range}")
        print(f"Reward Threshold: {spec.reward_threshold}")

    wandb.init(project="test1", config={"episode_num": 2000, "entropy_coefficient": 0.001})

    query_environment('CartPole-v1')
    env = gym.make('CartPole-v1')
    model = Actor_Critic(env)  #实例化Actor_Critic算法类
    reward = []
    for episode in range(2000):
        s_s = [env.reset()[0]]  #获取环境状态
        a_s = []
        log_prob_s = []
        rew_s = []
        done = False     #记录当前回合游戏是否结束
        ep_r = 0

        while not done:
            # 通过Actor_Critic算法对当前环境做出行动
            a, log_prob = model.get_action(s_s[-1])
            a_s.append(a)
            log_prob_s.append(log_prob)
            # 获得在做出a行动后的最新环境
            s_, rew, done, _, _ = env.step(a)
            s_s.append(s_)
            rew_s.append(rew)

            #计算当前reward
            ep_r += rew

        #训练模型
        model.learn(log_prob_s, s_s, done, rew_s, 0.001)

        #显示奖励
        reward.append(ep_r)
        wandb.log({"episode_reward":ep_r})
        print(f"episode:{episode} ep_r:{ep_r}")

    wandb.finish()
    plt.plot(reward)
    plt.show()
