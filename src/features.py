"""Builds the model's input numbers (features) for one payment at a time.

The same code is used for training and for the live checker,
so both see identical numbers.
"""
from collections import deque
from datetime import timedelta

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}

# Values used when a card has no earlier payments (worked out from the train data)
FILL_CARD_AVG_AMT = 47.52
FILL_AMT_VS_USUAL = 1.0
FILL_MINS_SINCE_LAST = 276.0

FEATURE_COLUMNS = (
    ["amt", "hour", "is_night"]
    + ["cat_" + c for c in CATEGORIES]
    + ["card_avg_amt", "amt_vs_usual", "recent_10min", "recent_60min",
       "mins_since_last", "is_new_merchant", "has_history"]
)


class CardHistory:
    """What we remember about one card's earlier payments."""

    def __init__(self):
        self.count = 0
        self.amt_sum = 0.0
        self.last_time = None
        self.merchants = set()
        self.recent_times = deque()  # times of payments in the last 60 minutes


class FeatureBuilder:
    """Keeps a memory of every card and turns each new payment into features."""

    def __init__(self):
        self.cards = {}

    def process(self, payment):
        """payment: dict with cc_num, merchant, category, amt and time (a Timestamp).
        Returns a dict of features, then adds the payment to the card's memory."""
        t = payment["time"]
        h = self.cards.setdefault(payment["cc_num"], CardHistory())

        # forget payments older than 60 minutes
        while h.recent_times and h.recent_times[0] < t - timedelta(minutes=60):
            h.recent_times.popleft()

        has_history = h.count > 0
        card_avg = h.amt_sum / h.count if has_history else FILL_CARD_AVG_AMT
        amt_vs_usual = payment["amt"] / card_avg if has_history else FILL_AMT_VS_USUAL
        mins_since_last = (
            (t - h.last_time).total_seconds() / 60 if has_history else FILL_MINS_SINCE_LAST
        )

        f = {
            "amt": payment["amt"],
            "hour": t.hour,
            "is_night": int(t.hour in NIGHT_HOURS),
            "card_avg_amt": card_avg,
            "amt_vs_usual": amt_vs_usual,
            "recent_10min": sum(1 for x in h.recent_times if x >= t - timedelta(minutes=10)),
            "recent_60min": len(h.recent_times),
            "mins_since_last": mins_since_last,
            "is_new_merchant": int(payment["merchant"] not in h.merchants),
            "has_history": int(has_history),
        }
        for c in CATEGORIES:
            f["cat_" + c] = int(payment["category"] == c)

        # only now remember this payment, so it never counts as its own history
        h.count += 1
        h.amt_sum += payment["amt"]
        h.last_time = t
        h.merchants.add(payment["merchant"])
        h.recent_times.append(t)
        return f
