


def main():

    initialize_Q()    

    while True:
        ts = select_ts_from_Q()
        effect = perform_action(ts)
        reward = measure_reward(effect)
        update_Q(ts, effect, reward)

    print('asdf')


def initialize_Q():
    pass


def select_ts_from_Q():
    pass


def perform_action(ts):
    pass


def measure_reward(effect):
    pass


def update_Q(ts, effect, reward):
    pass


if __name__ == '__main__':
    main()
