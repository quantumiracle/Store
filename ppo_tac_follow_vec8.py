"""
pure vector observation based learning: position of tactip and target
task: tactip following the cylinder to reach the ball target
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from gym_unity.envs import UnityEnv
import argparse
from PIL import Image
import time



# from env import Reacher


EP_MAX = 100000
EP_LEN = 99
N_WORKER = 1                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 10         # minimum batch size for updating PPO
UPDATE_STEP = 3            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
S_DIM, A_DIM, CHANNEL = 256, 2, 1       # state and action dimension
NUM_PINS = 91  #127
VS_DIM = 4*(2)  # dim of vector state
S_DIM_ALL =  S_DIM*S_DIM*CHANNEL
env_name = "./tac_follow_new"  # Name of the Unity environment binary to launch
# env = UnityEnv(env_name, worker_id=2, use_visual=False)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()



class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.vs = tf.placeholder(tf.float32, [None, VS_DIM], 'state')
        # self.tfs = tf.placeholder(tf.float32, [None, S_DIM, S_DIM, 1], 'state')
        self.tfs = self.vs


        # critic
        # encoded = self.encoder(self.tfs)  # convolutional encoder
        encoded = self.tfs
        l1 = tf.layers.dense(encoded, 100, tf.nn.relu)
        l2 = tf.layers.dense(l1, 100, tf.nn.relu)
        self.v = tf.layers.dense(l2, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    # def maxpool2d(x, k=2):
    #     # MaxPool2D wrapper
    #     return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                # s, a, r = data[:, :S_DIM_ALL], data[:, S_DIM_ALL: S_DIM_ALL + A_DIM], data[:, -1:]
                # s = s.reshape(-1, S_DIM,S_DIM,CHANNEL)
                s, a, r = data[:, :VS_DIM], data[:, VS_DIM: VS_DIM + A_DIM], data[:, -1:]

                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available
    
    def encoder(self, input):
        model = tf.keras.models.Sequential(name='encoder')
        model.add(Conv2D(filters=8, kernel_size=(2,2), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Conv2D(filters=4, kernel_size=(2,2), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        # model.add(Conv2D(filters=2, kernel_size=(2,2), padding='same', activation='relu'))
        # model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Flatten())
        return model(input)
        # c1 = tf.keras.layers.Conv2D(filters=4, kernel_size=(2,2), padding='same', activation='relu')
        # p1 = self.maxpool2d(c1, k=2)
        # c2 = tf.keras.layers.Conv2D(filters=4, kernel_size=(2,2), padding='same', activation='relu')
        # p1 = self.maxpool2d(c1, k=2)
        # tf.layers.flatten(p2)


        

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            # encoded = self.encoder(self.tfs)
            # print('latent dim: ', encoded.shape)
            encoded = self.tfs
            l1 = tf.layers.dense(encoded, 200, tf.nn.tanh, trainable=trainable)
            l2 = tf.layers.dense(l1, 200, tf.nn.tanh, trainable=trainable)
            action_scale = 2.0
            mu = action_scale * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            sigma +=1e-3 # without this line, 0 value sigma may cause NAN action
            # print('mu,sig: ', mu, sigma)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :] # np.newaxis is to increase dim, [] -> [[]]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -360, 360)

    def get_v(self, s):
        if s.ndim < 4: s = s[np.newaxis, :] 
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


    def save(self, path):
            self.saver.save(self.sess, path)

    def load(self, path):
            self.saver.restore(self.sess, path)




class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = UnityEnv(env_name, worker_id=wid, use_visual=False, use_both=True)

        # self.env=Reacher(render=True)
        self.ppo = GLOBAL_PPO


    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        step_set=[]
        epr_set=[]
        step=0
        while not COORD.should_stop():
            s, info= self.env.reset()
            s=s[:8]
            step+=1
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            self.pins_x=[]
            self.pins_y=[]
            self.pins_z=[]
            self.object_x=[]
            self.object_y=[]
            self.object_z=[]
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data

                a = self.ppo.choose_action(s)
                s_, r, done, info= self.env.step(a)
                # print(np.array(s_).shape)

                # plot pins
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)                    # normalize reward, find to be useful

                pins_x = s_[6::3]
                pins_z = s_[8::3]
                self.object_x.append(s_[0]) 
                self.object_z.append(s_[2])
                self.pins_x.append(pins_x)
                self.pins_z.append(pins_z)

                relative_x=pins_x-s_[0]
                relative_z=pins_z-s_[2]
                dis = (relative_x - (self.pins_x[0]-self.object_x[0]))**2 + (relative_z - (self.pins_z[0]-self.object_z[0]))**2
                min_idx = np.argmin(dis)
                max_idx = np.argmax(dis)
                # add relative position of the pin with smallest deformation
                s_ = np.append(s_[:6], relative_x[min_idx])
                s_ = np.append(s_, relative_z[min_idx])
                s = s_
                ep_r += r


                # print('minimal displacement idx: ', min_idx)

                GLOBAL_UPDATE_COUNTER += 1                      # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break


            if GLOBAL_EP%50==0 and GLOBAL_EP>0:
                self.ppo.save(model_path)

            reshape_pins_x = np.array(self.pins_x).transpose()
            reshape_pins_z = np.array(self.pins_z).transpose()
            plt.clf()
            for i in range(NUM_PINS):
                plt.subplot(411)
                plt.plot(np.arange(len(self.pins_x)), reshape_pins_x[i])
                plt.title('X-Position')
                plt.subplot(412)
                plt.plot(np.arange(len(self.pins_z)), reshape_pins_z[i])
                plt.title('Y-Position')  # although it's z, to match reality, use y
                plt.subplot(413)
                # plt.plot(np.arange(len(self.pins_x)), reshape_pins_x[i]-self.object_x)
                # plt.title('X-Relative')
                # plt.plot(np.arange(len(self.pins_x)), reshape_pins_x[i]-reshape_pins_x[i][0])
                # plt.title('X-Displacement')
                plt.plot(np.arange(len(self.pins_x)), (reshape_pins_x[i]-self.object_x)-(reshape_pins_x[i][0]-self.object_x[0]))
                plt.title('X-Displacement')

                plt.subplot(414)
                plt.plot(np.arange(len(self.pins_x)), (reshape_pins_z[i]-self.object_z)-(reshape_pins_z[i][0]-self.object_z[0]))
                plt.title('Y-Displacement')
                plt.xlabel('Time Step')
                plt.tight_layout()
            plt.savefig('./ppo_pins.png')
                    


            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)
            step_set.append(step)
            # print(step)
            epr_set.append(ep_r)
            if step % 10==0:  # plot every N episode; some error about main thread for plotting
                plt.clf()
                plt.plot(step_set,epr_set)
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                try:
                    plt.savefig('./tac_pins.png')
                except:
                    print('writing conflict!')
                



if __name__ == '__main__':
    model_path = './model/tac_pins'
    if args.train:
        time=time.time()
        GLOBAL_PPO = PPO()
        UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
        UPDATE_EVENT.clear()            # not update now
        ROLLING_EVENT.set()             # start to roll out
        workers = [Worker(wid=i) for i in range(N_WORKER)]
        
        GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
        GLOBAL_RUNNING_R = []
        COORD = tf.train.Coordinator()
        QUEUE = queue.Queue()           # workers putting data in this queue
        threads = []
        for worker in workers:          # worker threads
            t = threading.Thread(target=worker.work, args=())
            t.daemon = True             # kill the main thread, the sub-threads die as well
            t.start()                   # training
            threads.append(t)
        # add a PPO updating thread
        threads.append(threading.Thread(target=GLOBAL_PPO.update,))
        threads[-1].start()
        COORD.join(threads)

        GLOBAL_PPO.save(model_path)

    if args.test:
        env = UnityEnv(env_name, worker_id=np.random.randint(0,10), use_visual=False, use_both=True)
        env.reset()
        GLOBAL_PPO = PPO()
        GLOBAL_PPO.load(model_path)
        test_steps = 99
        test_episode = 10
        

        for _ in range(test_episode):
            s, info = env.reset()
            for t in range(test_steps):
                # env.render()
                s, r, done, info = env.step(GLOBAL_PPO.choose_action(s))
      


