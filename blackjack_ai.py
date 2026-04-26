import random
from collections import defaultdict

def draw_card():
    deck = [1,2,3,4,5,6,7,8,9,10,10,10,10]
    return random.choice(deck)

def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

def hand_value(hand):
    s = sum(hand)
    if usable_ace(hand):
        return s + 10
    return s

def is_bust(hand):
    return hand_value(hand) > 21

def get_state(player, dealer):
    return (hand_value(player), dealer[0], usable_ace(player))

def dealer_play(dealer):
    while hand_value(dealer) < 17:
        dealer.append(draw_card())
    return dealer

def generate_episode(policy, epsilon=0.1):
    player = [draw_card(), draw_card()]
    dealer = [draw_card(), draw_card()]

    episode = []

    while True:
        state = get_state(player, dealer)

        if random.random() < epsilon:
            action = random.choice([0,1])
        else:
            action = policy[state]

        episode.append((state, action))

        if action == 1:
            player.append(draw_card())
            if is_bust(player):
                return episode, -1
        else:
            break

    dealer_play(dealer)

    player_score = hand_value(player)
    dealer_score = hand_value(dealer)

    if dealer_score > 21 or player_score > dealer_score:
        return episode, 1
    elif player_score < dealer_score:
        return episode, -1
    else:
        return episode, 0

def train_mc(episodes=500_000):
    Q = defaultdict(lambda: [0.0, 0.0])
    returns = defaultdict(list)
    policy = defaultdict(lambda: random.choice([0,1]))

    for i in range(episodes):
        epsilon = max(0.01, 0.1 * (1 - i / episodes))
        episode, reward = generate_episode(policy, epsilon)

        visited = set()

        for state, action in episode:
            if (state, action) not in visited:
                returns[(state, action)].append(reward)
                Q[state][action] = sum(returns[(state, action)]) / len(returns[(state, action)])
                policy[state] = 0 if Q[state][0] >= Q[state][1] else 1
                visited.add((state, action))

        if (i+1) % 100000 == 0:
            print("Обучено:", i+1)

    return policy

def evaluate(policy, episodes=50000):
    wins = 0

    for _ in range(episodes):
        player = [draw_card(), draw_card()]
        dealer = [draw_card(), draw_card()]

        while True:
            state = get_state(player, dealer)
            action = policy[state]

            if action == 1:
                player.append(draw_card())
                if is_bust(player):
                    break
            else:
                break

        dealer_play(dealer)

        ps = hand_value(player)
        ds = hand_value(dealer)

        if ps <= 21 and (ds > 21 or ps > ds):
            wins += 1

    print("Win rate:", wins / episodes * 100)

if __name__ == "__main__":
    policy = train_mc()
    evaluate(policy)