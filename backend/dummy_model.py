def fake_fraud_probability(data):
    amount, hour, new_device, location_change, daily_count = data

    score = 0.0

    # Amount rule
    if amount > 20000:
        score += 0.4
    elif amount > 10000:
        score += 0.25
    elif amount > 5000:
        score += 0.15

    # Time rule (late night)
    if hour < 6 or hour > 22:
        score += 0.2

    # Device & location
    if new_device == 1:
        score += 0.15

    if location_change == 1:
        score += 0.15

    # Frequency
    if daily_count > 6:
        score += 0.2
    elif daily_count > 3:
        score += 0.1

    # Cap probability at 0.99
    return min(score, 0.99)
